import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from flask import Flask, request, render_template, jsonify, session
from pydub import AudioSegment
import soundfile as sf
from processing import process_audio
import requests
import json
from dotenv import load_dotenv  

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
    
    Args:
        transcription (str): The transcribed text to summarize
        
    Returns:
        dict: Contains success status and either summary or error message
    """
    try:
        api_key = os.environ.get('GEMINI_API_KEY')
        
        if not api_key:
            print("❌ Warning: GEMINI_API_KEY not found in environment variables.")
            return {
                'success': False,
                'error': 'Gemini API key not configured'
            }
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        prompt = (
            "Summarize the following transcribed and translated lecture recordings from a "
            "classroom. The lectures "
            "contain both English and Malayalam, but the provided text is already translated "
            "into English. Focus on key concepts, important explanations, and any technical terms "
            "covered. Ensure the summary is concise but retains the essential details of the topics "
            "discussed. Using these transcribed text, generate a detailed note for students to refer "
            "during learning sessions, these notes can also include information not mentioned during "
            "the class, include sample questions where possible. Do not use bold text or italics in your response\n\n"
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
        
        response = requests.post(
            f"{url}?key={api_key}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            response_data = response.json()
            
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                if 'content' in response_data['candidates'][0]:
                    content = response_data['candidates'][0]['content']
                    if 'parts' in content and len(content['parts']) > 0:
                        summary = content['parts'][0]['text']
                        return {
                            'success': True,
                            'summary': summary
                        }
            
            return {
                'success': False,
                'error': 'Unable to extract summary from API response'
            }
        
        else:
            print(f"❌ Gemini API error (status code {response.status_code}): {response.text}")
            return {
                'success': False,
                'error': f"API request failed with status code {response.status_code}"
            }
            
    except Exception as e:
        print(f"❌ Error calling Gemini API: {str(e)}")
        return {
            'success': False,
            'error': f"Error generating summary: {str(e)}"
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

    if file and allowed_file(file.filename):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        print(f"File saved at: {file_path}")
        
        if not os.path.exists(file_path):
            return jsonify({'error': '❌ File was not saved correctly'}), 500
            
        if os.path.getsize(file_path) == 0:
            return jsonify({'error': '❌ Uploaded file is empty'}), 400

        processing_result = process_audio(file_path, UPLOAD_FOLDER)
        
        if 'success' in processing_result and processing_result['success']:
            # Store transcription results
            transcription = processing_result['transcription']
            session['message'] = '✅ File uploaded and processed successfully'
            session['transcription'] = transcription
            session['output_file'] = processing_result['output_file']
            
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
    """Generate notes from the transcription using Gemini API."""

    transcription = session.get('transcription', None)
    
    if not transcription:
        return jsonify({
            'success': False,
            'error': 'No transcription available. Please process an audio file first.'
        }), 400
    
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

if __name__ == '__main__':
    app.run(debug=False)
