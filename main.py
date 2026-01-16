import os
import logging
import time
import random
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from flask import Flask, request, render_template, jsonify, session, send_file
from pydub import AudioSegment
import soundfile as sf
from processing import process_audio
import requests
import json
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from io import BytesIO

load_dotenv()

# Configure other logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = 'dualdecode_secret_key'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'speechprocessing')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"Upload folder created at: {UPLOAD_FOLDER}")

ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    """Check if the uploaded file has a valid WAV extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def split_audio(file_path, output_folder="speechprocessing"):
    """Splits a WAV audio file into 50-second segments."""
    try:
        os.makedirs(output_folder, exist_ok=True)  

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ Error: File {file_path} not found")
            return False

        # Print file info for debugging
        print(f"Processing file: {file_path}")
        print(f"File size: {os.path.getsize(file_path)} bytes")

        try:
            audio = AudioSegment.from_file(file_path, format="wav")
            print(f"Successfully loaded audio: {len(audio)}ms duration, {audio.channels} channels, {audio.frame_rate}Hz")
        except Exception as e:
            print(f"❌ Error loading audio with pydub: {str(e)}")

            try:
                import soundfile as sf
                data, samplerate = sf.read(file_path)
                from pydub.audio_segment import AudioSegment as PydubAudioSegment
                audio = PydubAudioSegment(
                    data.tobytes(),
                    frame_rate=samplerate,
                    sample_width=data.dtype.itemsize,
                    channels=1 if len(data.shape) == 1 else data.shape[1]
                )
                print(f"Loaded with soundfile backup method: {len(audio)}ms")
            except Exception as backup_error:
                print(f"❌ Both loading methods failed. Backup error: {str(backup_error)}")
                return False

        duration = len(audio)  # Duration in milliseconds
        segment_length = 50 * 1000  # 50 seconds in milliseconds
        count = 1
        segment_paths = []

        for start in range(0, duration, segment_length):
            end = min(start + segment_length, duration)
            segment = audio[start:end]

            segment_filename = f"unprocessedsplitaudio{count}.wav"
            segment_path = os.path.join(output_folder, segment_filename)
            
            try:
                segment.export(segment_path, format="wav")
                print(f"✅ Saved: {segment_path}")
                segment_paths.append(segment_path)
            except Exception as export_error:
                print(f"❌ Error exporting segment {count}: {str(export_error)}")
            
            count += 1

        return segment_paths if segment_paths else False

    except Exception as e:
        print(f"❌ Error processing audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return False  # Failure

def generate_summary_with_gemini(transcription):
    """
    Generate a summary of the transcribed text using Gemini API.
    
    IMPORTANT: This function is called ONCE per session after all audio segments 
    have been combined into a single transcription. It does NOT loop through segments.
    
    Rate Limiting Strategy:
    - Uses exponential backoff with jitter to handle 429 (rate limit) errors
    - Handles RESOURCE_EXHAUSTED status (free tier quota exceeded)
    - Respects Gemini API rate limits to avoid service disruption
    
    Args:
        transcription (str): The combined transcribed text from all audio segments
        
    Returns:
        dict: Contains success status and either summary or error message
    """
    MAX_RETRIES = 5  # More retries for quota exhaustion
    BASE_WAIT_TIME = 2  # Start with 2 seconds
    
    try:
        api_key = os.environ.get('GEMINI_API_KEY')
        
        if not api_key:
            print("❌ FATAL: GEMINI_API_KEY not found in environment variables.")
            return {
                'success': False,
                'error': 'Gemini API key not configured'
            }
        
        print(f"\n✅ API Key found: {api_key[:10]}...{api_key[-5:]}")
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        
        prompt = (
            "Summarize the following transcribed and translated lecture recordings from a "
            "classroom. The lectures "
            "contain both English and Malayalam, but the provided text is already translated "
            "into English. Focus on key concepts, important explanations, and any technical terms "
            "covered. Ensure the summary is concise but retains the essential details of the topics "
            "discussed. Using these transcribed text, generate a detailed note for students to refer "
            "during learning sessions, these notes can also include information not mentioned during "
            "the class, include sample questions where possible. Do not use bold text or italics in your response. Do not use precontext. write the topic of the class lecture first.\n\n"
            f"Transcribed text:\n{transcription}"
        )
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        print(f"\n{'='*70}")
        print(f"🔍 GEMINI API TEST & DEBUG")
        print(f"{'='*70}")
        print(f"✅ API Endpoint: {url}")
        print(f"✅ Transcription: {len(transcription)} characters")
        print(f"{'='*70}\n")
        
        # Retry loop with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                print(f"📤 [Attempt {attempt + 1}/{MAX_RETRIES}] Calling Gemini API...")
                
                response = requests.post(
                    f"{url}?key={api_key}",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=30
                )
                
                print(f"📊 Response Status: {response.status_code} ({response.reason})")
                
                # Parse response
                try:
                    response_data = response.json()
                    error_data = response_data.get('error', {})
                    status = error_data.get('status', 'UNKNOWN')
                    message = error_data.get('message', 'No message')
                    
                    print(f"📋 Full Response ({len(str(response_data))} chars):")
                    print(json.dumps(response_data, indent=2)[:1200])
                    
                    if status:
                        print(f"\n❌ Error Status: {status}")
                        print(f"   Message: {message[:200]}")
                
                except Exception as parse_err:
                    print(f"📋 Response Text:\n{response.text[:800]}")
                    response_data = None
                
                # SUCCESS: 200 OK
                if response.status_code == 200:
                    if 'candidates' in response_data and len(response_data['candidates']) > 0:
                        if 'content' in response_data['candidates'][0]:
                            content = response_data['candidates'][0]['content']
                            if 'parts' in content and len(content['parts']) > 0:
                                summary = content['parts'][0]['text']
                                print(f"\n✅ SUCCESS: Summary generated!")
                                print(f"   Length: {len(summary)} characters\n")
                                return {
                                    'success': True,
                                    'summary': summary
                                }
                    
                    print(f"\n⚠️  Unexpected: 200 OK but no content extracted")
                    print(f"   Response keys: {response_data.keys() if response_data else 'None'}\n")
                    return {
                        'success': False,
                        'error': 'API returned 200 but no summary found'
                    }
                
                # 429 / RESOURCE_EXHAUSTED: Rate limit or quota exceeded
                if response.status_code == 429:
                    error_data = response_data.get('error', {}) if response_data else {}
                    status = error_data.get('status', '')
                    message = error_data.get('message', '')
                    
                    print(f"\n⚠️  [429] Rate Limit / Quota Issue")
                    print(f"   Status: {status}")
                    
                    # Extract retry delay from response
                    retry_delay = 1
                    if response_data and 'error' in response_data:
                        details = response_data['error'].get('details', [])
                        for detail in details:
                            if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                                retry_str = detail.get('retryDelay', '1s')
                                try:
                                    retry_delay = float(retry_str.rstrip('s'))
                                except:
                                    retry_delay = 1
                    
                    if attempt < MAX_RETRIES - 1:
                        # Use API's suggested retry delay + exponential backoff
                        wait_time = max(retry_delay, BASE_WAIT_TIME * (2 ** attempt)) + random.uniform(0.5, 2)
                        print(f"⏳ Waiting {wait_time:.1f}s before retry...")
                        print(f"   (API suggested: {retry_delay}s, Exponential: {BASE_WAIT_TIME * (2 ** attempt)}s)\n")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"\n❌ FINAL: Rate limit/quota exceeded after {MAX_RETRIES} attempts")
                        print(f"   This likely means: Free tier quota exhausted or too many requests")
                        print(f"   Action: Upgrade to paid plan or wait for quota reset\n")
                        
                        if 'RESOURCE_EXHAUSTED' in message or 'quota' in message.lower():
                            return {
                                'success': False,
                                'error': 'API quota exhausted. Upgrade to paid plan: https://ai.google.dev'
                            }
                        else:
                            return {
                                'success': False,
                                'error': 'API rate limit exceeded. Please try again in a few minutes.'
                            }
                
                # 500+ Server errors
                if response.status_code >= 500:
                    print(f"\n❌ Server Error {response.status_code}")
                    if attempt < MAX_RETRIES - 1:
                        wait_time = BASE_WAIT_TIME * (2 ** attempt) + random.uniform(0.5, 2)
                        print(f"⏳ Waiting {wait_time:.1f}s before retry...\n")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {
                            'success': False,
                            'error': f'API server error {response.status_code}. Try again later.'
                        }
                
                # Other HTTP errors
                if response.status_code != 200:
                    print(f"\n❌ HTTP Error {response.status_code}: {response.reason}")
                    if response_data:
                        print(f"   Error: {response_data.get('error', {}).get('message', 'Unknown')[:150]}\n")
                    return {
                        'success': False,
                        'error': f"API error {response.status_code}: {response.reason}"
                    }
                
            except requests.exceptions.Timeout as timeout_err:
                print(f"\n❌ Timeout Error: {str(timeout_err)[:100]}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = BASE_WAIT_TIME * (2 ** attempt) + random.uniform(0.5, 2)
                    print(f"⏳ Waiting {wait_time:.1f}s before retry...\n")
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        'success': False,
                        'error': 'API request timeout. Try with shorter transcription.'
                    }
            
            except requests.exceptions.ConnectionError as conn_error:
                print(f"\n❌ Connection Error: {str(conn_error)[:100]}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = BASE_WAIT_TIME * (2 ** attempt) + random.uniform(0.5, 2)
                    print(f"⏳ Waiting {wait_time:.1f}s before retry...\n")
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        'success': False,
                        'error': 'Network error. Check your internet connection.'
                    }
            
            except Exception as retry_error:
                print(f"\n❌ Unexpected Error: {type(retry_error).__name__}")
                print(f"   Details: {str(retry_error)[:100]}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = BASE_WAIT_TIME * (2 ** attempt) + random.uniform(0.5, 2)
                    print(f"⏳ Waiting {wait_time:.1f}s before retry...\n")
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        'success': False,
                        'error': f'Unexpected error: {str(retry_error)[:100]}'
                    }
        
        print(f"\n❌ FINAL: All {MAX_RETRIES} retry attempts exhausted\n")
        return {
            'success': False,
            'error': 'Failed after all retry attempts. API may be unavailable.'
        }
            
    except Exception as e:
        print(f"\n❌ OUTER EXCEPTION: {type(e).__name__}")
        print(f"   Details: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        print()
        return {
            'success': False,
            'error': f"Error: {str(e)[:100]}"
        }

@app.route('/')
def index():
    """Render the landing page."""
    return render_template('landing.html')

@app.route('/upload')
def upload_page():
    """Render the upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    """Handle audio file uploads, split the audio, and process it for transcription."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Get language from form, default to Malayalam
    language = request.form.get('language', 'malayalam').lower()
    if language not in ['malayalam', 'hindi']:
        language = 'malayalam'
    
    print(f"✅ Language selected: {language}")

    if file and allowed_file(file.filename):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        print(f"File saved at: {file_path}")
        
        if not os.path.exists(file_path):
            return jsonify({'error': '❌ File was not saved correctly'}), 500
            
        if os.path.getsize(file_path) == 0:
            return jsonify({'error': '❌ Uploaded file is empty'}), 400

        processing_result = process_audio(file_path, UPLOAD_FOLDER, language)
        
        if 'success' in processing_result and processing_result['success']:
            # Store transcription results and language choice
            transcription = processing_result['transcription']
            session['message'] = '✅ File uploaded and processed successfully'
            session['transcription'] = transcription
            session['output_file'] = processing_result['output_file']
            session['language'] = language
            
            session['summary'] = None
            session['summary_error'] = None
            
            return jsonify({
                'success': True,
                'redirect': '/results'
            }), 200
        else:
            error_message = processing_result.get('error', 'Unknown error')
            session['message'] = f"❌ File processing failed: {error_message}"
            session['transcription'] = None
            session['output_file'] = None
            
            return jsonify({
                'success': False,
                'redirect': '/results',
                'error': error_message
            }), 200

    session['message'] = '❌ Invalid file format. Only WAV is supported.'
    return jsonify({
        'success': False,
        'redirect': '/results'
    }), 200

@app.route('/generate-notes', methods=['POST'])
def generate_notes():
    """
    Generate notes from the SINGLE COMBINED transcription using Gemini API.
    
    IMPORTANT: This endpoint is called ONCE when the user clicks "Create Notes".
    The transcription in the session is already a combination of all audio segments.
    NO LOOPS - NO MULTIPLE API CALLS.
    
    The API is called a single time with all transcription data.
    """

    transcription = session.get('transcription', None)
    
    if not transcription:
        return jsonify({
            'success': False,
            'error': 'No transcription available. Please process an audio file first.'
        }), 400
    
    # Single API call - NOT in a loop
    summary_result = generate_summary_with_gemini(transcription)
    
    if summary_result['success']:
        session['summary'] = summary_result['summary']
        session['summary_error'] = None
        
        return jsonify({
            'success': True,
            'summary': summary_result['summary']
        }), 200
    else:
        session['summary'] = None
        session['summary_error'] = summary_result['error']
        
        return jsonify({
            'success': False,
            'error': summary_result['error']
        }), 500

@app.route('/results')
def results_page():
    """Display the results of audio processing."""
    message = session.get('message', 'No processing information available')
    transcription = session.get('transcription', None)
    output_file = session.get('output_file', None)
    summary = session.get('summary', None)
    summary_error = session.get('summary_error', None)
    
    return render_template('results.html', 
                          message=message, 
                          transcription=transcription, 
                          output_file=output_file,
                          summary=summary,
                          summary_error=summary_error)

@app.route('/segments', methods=['GET'])
def list_segments():
    """List all available audio segments."""
    try:
        audio_files = [f for f in os.listdir(UPLOAD_FOLDER) 
                      if f.startswith('unprocessedsplitaudio') and f.endswith('.wav')]
        
        # Sort files numerically
        audio_files.sort(key=lambda x: int(os.path.basename(x).replace('unprocessedsplitaudio', '').replace('.wav', '')))
        
        if not audio_files:
            return jsonify({'message': 'No audio segments found'}), 404
            
        return jsonify({
            'message': f'Found {len(audio_files)} audio segments',
            'segments': audio_files
        }), 200
            
    except Exception as e:
        print(f"❌ Error listing segments: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'❌ Error listing segments: {str(e)}'}), 500

@app.route('/download-transcription', methods=['GET'])
def download_transcription():
    """Download transcription as PDF."""
    try:
        transcription = session.get('transcription', None)
        
        if not transcription:
            return jsonify({'error': 'No transcription available'}), 400
        
        # Create PDF in memory
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#D8C1E8',  # Pastel purple
            spaceAfter=30,
            alignment=1,  # Center alignment
        )
        
        content_style = ParagraphStyle(
            'CustomContent',
            parent=styles['BodyText'],
            fontSize=11,
            leading=14,
            alignment=4,  # Justify alignment
        )
        
        # Add title
        title = Paragraph("Lecture Transcription", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        # Add metadata
        date_para = Paragraph(f"<b>Generated on:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        elements.append(date_para)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add transcription text with word wrapping
        transcription_lines = transcription.split('\n')
        for line in transcription_lines:
            if line.strip():
                para = Paragraph(line, content_style)
                elements.append(para)
            else:
                elements.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        doc.build(elements)
        
        # Get PDF data
        pdf_buffer.seek(0)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'transcription_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
    
    except Exception as e:
        print(f"❌ Error generating PDF: {str(e)}")
        return jsonify({'error': f'Error generating PDF: {str(e)}'}), 500

@app.route('/download-summary', methods=['GET'])
def download_summary():
    """Download summary/notes as PDF."""
    try:
        summary = session.get('summary', None)
        
        if not summary:
            return jsonify({'error': 'No summary available'}), 400
        
        # Extract topic from the first line/sentence of the summary
        summary_lines = summary.split('\n')
        topic = "Study Notes"
        topic_line_index = 0
        
        # Find the first non-empty line as the topic
        for i, line in enumerate(summary_lines):
            if line.strip():
                topic = line.strip()
                # Limit topic length for PDF heading
                if len(topic) > 100:
                    topic = topic[:97] + "..."
                topic_line_index = i
                break
        
        # Create PDF in memory
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#D8C1E8',  # Pastel purple
            spaceAfter=30,
            alignment=1,  # Center alignment
            bold=True,
        )
        
        content_style = ParagraphStyle(
            'CustomContent',
            parent=styles['BodyText'],
            fontSize=11,
            leading=14,
            alignment=4,  # Justify alignment
        )
        
        # Add topic as title
        title = Paragraph(topic, title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        # Add metadata
        date_para = Paragraph(f"<b>Generated on:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        elements.append(date_para)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add summary text with word wrapping, starting from the second line if topic was extracted
        for i, line in enumerate(summary_lines):
            # Skip the first line if it was used as topic
            if i == topic_line_index and i == 0:
                continue
            
            if line.strip():
                para = Paragraph(line, content_style)
                elements.append(para)
            else:
                elements.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        doc.build(elements)
        
        # Get PDF data
        pdf_buffer.seek(0)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'study_notes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
    
    except Exception as e:
        print(f"❌ Error generating PDF: {str(e)}")
        return jsonify({'error': f'Error generating PDF: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=False)
