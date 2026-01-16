import torch
from transformers import pipeline, WhisperProcessor
import soundfile as sf
import numpy as np
import os

def process_audio(audio_file_path, output_dir=None):
    """
    Process audio file and return transcription
    
    Args:
        audio_file_path: Path to the audio file to process
        output_dir: Directory to save chunks and transcription (optional)
        
    Returns:
        dict: Contains transcription text and path to saved file
    """
    # Configure CUDA for PyTorch
    if torch.cuda.is_available():
        device = "cuda"
        # Print detailed GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        print(f"✅ GPU enabled: {gpu_name} with {gpu_mem:.2f}GB memory")
        # Set CUDA device to avoid OOM errors
        torch.cuda.set_device(0)
    else:
        device = "cpu"
        print("⚠️ CUDA not available - using CPU instead")
    
    print(f"Using device: {device}")

    # Set default output directory if none provided
    if output_dir is None:
        output_dir = os.path.dirname(audio_file_path)
        
    # Check if file exists
    if not os.path.exists(audio_file_path):
        return {"success": False, "error": f"Audio file not found: {audio_file_path}"}

    try:
        # Read the audio file
        data, samplerate = sf.read(audio_file_path)
        
        # Function to split audio into 1-minute chunks
        def split_audio(data, samplerate, chunk_duration_s=60):
            chunk_samples = chunk_duration_s * samplerate
            num_chunks = len(data) // chunk_samples
            chunks = []

            for i in range(num_chunks):
                chunks.append(data[i * chunk_samples:(i + 1) * chunk_samples])

            # Last chunk (if smaller than 1 minute)
            if len(data) % chunk_samples != 0:
                chunks.append(data[num_chunks * chunk_samples:])

            return chunks

        # Split audio into 1-minute chunks
        chunks = split_audio(data, samplerate)

        # Create an output folder for storing chunks
        output_folder = os.path.join(output_dir, "audio_chunks")
        os.makedirs(output_folder, exist_ok=True)

        # Save the chunks to separate files
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(output_folder, f"chunk_{i+1}.wav")
            sf.write(chunk_file, chunk, samplerate)
            chunk_files.append(chunk_file)
            print(f"Saved chunk {i+1} as {chunk_file}")

        # Load the Whisper Large v3 model and processor
        model_name = "openai/whisper-large-v3"
        processor = WhisperProcessor.from_pretrained(model_name)
        
        # Explicitly set device mapping and configure pipeline with appropriate device
        if torch.cuda.is_available():
            model = pipeline(
                "automatic-speech-recognition", 
                model=model_name, 
                device=0,  # Use first CUDA device
                torch_dtype=torch.float16  # Use fp16 for efficiency on GPU
            )
        else:
            model = pipeline("automatic-speech-recognition", model=model_name, device=-1)

        # Translate each chunk into English using Whisper
        translated_texts = []

        for chunk_file in chunk_files:
            # Set the language to Malayalam using forced_decoder_ids
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="malayalam")

            # Transcribe the audio chunk with Whisper
            transcription = model(chunk_file, return_timestamps=True, generate_kwargs={"forced_decoder_ids": forced_decoder_ids})

            # Extract and store transcribed text
            translated_text = transcription["text"]
            translated_texts.append(translated_text)

            print(f"Transcription for {chunk_file}:")
            print(translated_text)

        # Combine transcriptions from all chunks
        full_transcription = " ".join(translated_texts)

        # Save the full transcription to a text file
        output_file = os.path.join(output_dir, "full_transcription.txt")
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(full_transcription)

        print(f"Full Transcription (English) saved to: {output_file}")
        
        return {
            "success": True,
            "transcription": full_transcription,
            "output_file": output_file
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

# Only run the function if script is executed directly (not imported)
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    
    # Create a file dialog to select the audio file
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    audio_file = filedialog.askopenfilename(title="Select an Audio File", filetypes=[("WAV files", "*.wav")])
    
    if not audio_file:
        print("No file selected. Exiting.")
        exit()
        
    result = process_audio(audio_file)
    print(result)
