import openai
import sounddevice as sd
import numpy as np
import termios
import fcntl
import os
import time
import sys
from datetime import datetime
from pynput import keyboard
import threading
import queue
from scipy.io import wavfile
from rapidfuzz import fuzz

# Initialize OpenAI client (you'll need to set your API key as an environment variable)
client = openai.OpenAI()

last_file = None
session_dir = "audio_sessions"


def extract_keyword(text):
    prompt = f"""
    Extract a single keyword that best represents the main subject of the following text. 
    The keyword must be a simple, one-word term (e.g., history, reminders, meetings).
    
    Text: {text}
    
    Keyword:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().lower()


def get_best_matching_directory(keyword):
    directories = [d for d in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, d))]
    if not directories:
        return keyword
    
    best_match = max(directories, key=lambda x: fuzz.ratio(keyword, x))
    if fuzz.ratio(keyword, best_match) > 80:  # If similarity is over 80%
        return best_match
    return keyword


def process_audio(continue_last=False):
    global last_file
    audio = record_audio()

    print("Transcribing audio...")
    raw_text = transcribe_audio(audio)

    print("Formatting idea...")
    formatted_idea = format_idea(raw_text)

    keyword = extract_keyword(raw_text)
    print(f"Extracted keyword: {keyword}")

    directory = get_best_matching_directory(keyword)
    directory_path = os.path.join(session_dir, directory)
    os.makedirs(directory_path, exist_ok=True)

    curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"note_{keyword}_{curr_date}.txt"
    filepath = os.path.join(directory_path, filename)

    if continue_last and last_file:
        with open(last_file, "a") as f:
            f.write(f"\nContinuation ({curr_date}):\n\n")
            f.write(f"Raw Transcription:\n{raw_text}\n\n")
            f.write(f"Formatted Idea:\n{formatted_idea}\n")
        print(f"Continued note in {os.path.basename(last_file)}")
    else:
        with open(filepath, "w") as f:
            f.write(f"Raw Transcription:\n{raw_text}\n\n")
            f.write(f"Formatted Idea:\n{formatted_idea}\n")
        last_file = filepath
        print(f"New note saved as {filename} in directory {directory}")


def on_press(key):
    global should_stop
    if key == keyboard.Key.enter:
        should_stop = True
        return False  # Stop listener


def record_audio():
    global should_stop
    should_stop = False
    sample_rate = 44100
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
        print("Recording... Press Enter to stop.")
        while not should_stop:
            sd.sleep(100)

    listener.join()
    print("Recording stopped.")

    audio_data = []
    while not q.empty():
        audio_data.append(q.get())

    return np.concatenate(audio_data)


def transcribe_audio(audio):
    temp_file = "temp_audio.wav"
    wavfile.write(temp_file, 44100, audio)

    with open(temp_file, "rb") as file:
        transcription = client.audio.transcriptions.create(model="whisper-1", file=file)

    os.remove(temp_file)
    return transcription.text


def format_idea(text):
    prompt = f"""
    Please format the following transcribed audio into a clear, readable structure:
    1. Clean up the text, fixing any grammatical errors and filling in obvious missing words.
    2. Organize the main points into a bullet-point list.
    3. Provide a brief summary (2-3 sentences) of the entire idea at the end.
    4. Do not add a second main point if 1 is sufficient.

    Original transcription: {text}

    Format your response as follows:
    Cleaned Text:
    [Insert cleaned and grammatically correct version of the full text here]

    Main Points:
    • [Main point 1]
    • [Main point 2]
    • [...]

    Summary:
    [Insert 2-3 sentence summary here]
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# def process_audio(continue_last=False):
#     global last_file
#     audio = record_audio()

#     print("Transcribing audio...")
#     raw_text = transcribe_audio(audio)

#     print("Formatting idea...")
#     formatted_idea = format_idea(raw_text)

#     # Save to file
#     session_dir = "audio_sessions"
#     os.makedirs(session_dir, exist_ok=True)

#     if continue_last and last_file:
#         filepath = last_file
#         mode = "a"  # append mode
#         print(f"Continuing last note in {os.path.basename(filepath)}")
#     else:
#         curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"idea_{curr_date}.txt"
#         filepath = os.path.join(session_dir, filename)
#         mode = "w"  # write mode
#         print(f"Creating new note: {filename}")

#     with open(filepath, mode) as f:
#         f.write(f"{'Continuation:' if continue_last else 'New Entry:'}\n\n")
#         f.write(f"Raw Transcription:\n{raw_text}\n\n")
#         f.write(f"Formatted Idea:\n{formatted_idea}\n")

#     last_file = filepath
#     print(
#         f"Idea {'appended to' if continue_last else 'saved in'} {os.path.basename(filepath)}"
#     )


def clear_input_buffer():
    # Get the file descriptor of standard input (stdin)
    fd = sys.stdin.fileno()

    # Get the current attributes of the file descriptor
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)

    # Temporarily set it to non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    try:
        # Read and discard all available input
        while sys.stdin.read(1):
            pass
    except IOError:
        pass
    finally:
        # Restore the original attributes
        fcntl.fcntl(fd, fcntl.F_SETFL, flags)


# def main():
#     global last_file
#     while True:
#         clear_input_buffer()
#         user_input = input(
#             "Press Enter to start recording, 'C' to continue last note, or 'Q' to quit: "
#         ).lower()
#         if user_input == "q":
#             break
#         elif user_input == "c":
#             if last_file:
#                 process_audio(continue_last=True)
#             else:
#                 print("No previous note to continue. Starting a new recording.")
#                 process_audio()
#         else:
#             process_audio()
#         print("Waiting for next command...")
#         time.sleep(1)  # Add a small delay


def main():
    global last_file
    while True:
        clear_input_buffer()
        user_input = input(
            "Press Enter to start recording, 'C' to continue last note, or 'Q' to quit: "
        ).lower()
        if user_input == "q":
            break
        elif user_input == "c":
            if last_file:
                process_audio(continue_last=True)
            else:
                print("No previous note to continue. Starting a new recording.")
                process_audio()
        else:
            process_audio()
        print("Waiting for next command...")
        time.sleep(1)  # Add a small delay


if __name__ == "__main__":
    main()
