import os
import pyaudio
import wave
import io
import time
import threading
from groq import Groq
import tkinter as tk
from tkinter import messagebox
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# ==========================
# Configuration and Setup
# ==========================
# Retrieve Groq API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("GROQ_API_KEY environment variable not set.")

# Initialize Groq client
client = Groq(api_key=groq_api_key)
whisper_model = "whisper-large-v3-turbo"
llama_model = "gemma2-9b-it"

# Audio Configuration
CHUNK = 1024            # Number of frames per buffer
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1            # Mono audio
RATE = 16000            # Sample rate for Whisper

# Initialize PyAudio
p = pyaudio.PyAudio()

# Global Variables
is_recording = False
frames = []
stream = None
recording_start_time = 0
exit_event = threading.Event()  # Event to signal threads to exit
record_lock = threading.Lock()   # Lock to synchronize access to recording state

# ==========================
# Helper Functions
# ==========================

def translate_text(text):
    """Translates the given text into English using the specified Llama model."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"""You are a professional translator. 
Your purpose is only to give translation from a text into English, 
the text is driven from speech so keep this context for better translation result due to some expressions that are 
not translated to text. Be accurate and don't say anything else no matter what. Here is the text: '{text}' """
            }],
            model=llama_model,
        )
        translated_text = chat_completion.choices[0].message.content.strip()
        return translated_text
    except Exception as e:
        print(f"Error in translation: {e}")
        return None

def process_audio(audio_buffer):
    """Processes the recorded audio: transcribes and translates it."""
    api_start_time = time.time()
    try:
        # Ensure the audio_buffer has data
        audio_data = audio_buffer.getvalue()
        if not audio_data:
            print("Recorded audio is empty. Skipping processing.")
            return

        print("Sending audio for transcription...")
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", audio_data),
            model=whisper_model,
        )
        api_end_time = time.time()
        api_duration = api_end_time - api_start_time

        if not transcription or not hasattr(transcription, 'text'):
            print("Transcription failed or returned empty.")
            return

        print(f"You said: {transcription.text}")
        print(f"API call duration for transcription: {api_duration:.2f} seconds")

        # Translate the transcription
        translation_start_time = time.time()
        translated_text = translate_text(transcription.text)
        translation_end_time = time.time()
        translation_duration = translation_end_time - translation_start_time

        if translated_text:
            print(f"Translated text: {translated_text}")
            print(f"API call duration for translation: {translation_duration:.2f} seconds")
        else:
            print("Translation failed or returned empty.")

        # Print total processing time
        total_duration = api_duration + translation_duration
        print(f"Total processing duration for this segment: {total_duration:.2f} seconds")

    except Exception as e:
        # Handle specific API errors
        if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            status_code = e.response.status_code
            error_message = e.response.json().get('error', {}).get('message', 'Unknown error')
            print(f"Error in transcription or translation: Error code: {status_code} - {error_message}")
        else:
            print(f"Error in transcription or translation: {e}")
    finally:
        audio_buffer.close()

def start_recording():
    """Starts recording audio from the microphone."""
    global is_recording, frames, stream, recording_start_time
    with record_lock:
        if is_recording:
            print("Already recording. Ignoring start request.")
            messagebox.showwarning("Warning", "Recording is already in progress.")
            return
        try:
            print("Recording started...")
            is_recording = True
            frames = []
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
            recording_start_time = time.time()
            status_label.config(text="Recording...")
        except Exception as e:
            print(f"Error starting recording: {e}")
            messagebox.showerror("Error", f"Failed to start recording: {e}")
            is_recording = False

def stop_recording():
    """Stops recording audio and initiates processing of the recorded data."""
    global is_recording, frames, stream
    with record_lock:
        if not is_recording:
            print("Not currently recording. Ignoring stop request.")
            messagebox.showwarning("Warning", "No recording is in progress.")
            return
        try:
            print("Recording stopped.")
            is_recording = False
            stream.stop_stream()
            stream.close()
            recording_end_time = time.time()
            duration = recording_end_time - recording_start_time
            print(f"Recording duration: {duration:.2f} seconds")
            status_label.config(text=f"Recorded {duration:.2f} seconds.")
        except Exception as e:
            print(f"Error stopping recording: {e}")
            messagebox.showerror("Error", f"Failed to stop recording: {e}")
            return
        finally:
            stream = None  # Ensure stream is set to None after closing

    # Write frames to an in-memory WAV file
    try:
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        audio_buffer.seek(0)

        # Optional: Verify the WAV file structure by attempting to read it
        try:
            with wave.open(audio_buffer, "rb") as wf_check:
                wf_check.getnframes()  # This will raise an error if WAV is malformed
        except wave.Error as e:
            print(f"WAV file is malformed: {e}. Skipping processing.")
            messagebox.showerror("Error", f"Recorded audio is malformed: {e}")
            return
    except Exception as e:
        print(f"Error writing audio buffer: {e}")
        messagebox.showerror("Error", f"Failed to write audio buffer: {e}")
        return

    # Process the audio in a separate thread
    processing_thread = threading.Thread(target=process_audio, args=(audio_buffer,))
    processing_thread.start()

def record_audio():
    """Continuously records audio data when recording is active."""
    global frames, stream
    while not exit_event.is_set():
        with record_lock:
            if is_recording and stream is not None:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    print(f"Error reading audio stream: {e}")
                    stop_recording()
        time.sleep(0.01)  # Small sleep to prevent high CPU usage

def on_closing():
    """Handles the window closing event."""
    if is_recording:
        if messagebox.askokcancel("Quit", "Recording is in progress. Do you want to stop and exit?"):
            stop_recording()
        else:
            return
    exit_event.set()
    root.destroy()

# ==========================
# GUI Setup
# ==========================

# Initialize the main window
root = tk.Tk()
root.title("Audio Recorder")
root.geometry("300x150")
root.resizable(False, False)

# Status label
status_label = tk.Label(root, text="Idle", font=("Helvetica", 12))
status_label.pack(pady=10)

# Start Recording Button
start_button = tk.Button(root, text="Start Recording", command=start_recording, width=20, bg="green", fg="white")
start_button.pack(pady=5)

# Stop Recording Button
stop_button = tk.Button(root, text="Stop Recording", command=stop_recording, width=20, bg="red", fg="white")
stop_button.pack(pady=5)

# Handle window close
root.protocol("WM_DELETE_WINDOW", on_closing)

# ==========================
# Audio Recording Thread
# ==========================

audio_thread = threading.Thread(target=record_audio, daemon=True)
audio_thread.start()

# ==========================
# Main Execution
# ==========================

def main():
    """Starts the GUI and manages the application lifecycle."""
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Exiting program via KeyboardInterrupt.")
    finally:
        # Ensure streams are closed and PyAudio is terminated
        with record_lock:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as e:
                    print(f"Error during stream closure: {e}")
        p.terminate()
        print("PyAudio terminated.")

if __name__ == "__main__":
    main()
