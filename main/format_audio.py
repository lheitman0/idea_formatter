import openai
import sounddevice as sd
import numpy as np
import os
import time
import sys
from datetime import datetime
from pynput import keyboard
import threading
import queue
import fcntl
from scipy.io import wavfile
import json

# Initialize OpenAI client (you'll need to set your API key as an environment variable)
client = openai.OpenAI()

desktop_path = os.path.expanduser("~/Desktop")
notes_dir = os.path.join(desktop_path, "notes")
tag_index_file = os.path.join(notes_dir, "tag_index.json")
last_file = None


def get_existing_categories():
    return [
        d for d in os.listdir(notes_dir) if os.path.isdir(os.path.join(notes_dir, d))
    ]


def add_new_category(category):
    category_path = os.path.join(notes_dir, category)
    os.makedirs(category_path, exist_ok=True)
    print(f"New category created: {category}")


def determine_category_and_tags(text):
    existing_categories = get_existing_categories()
    categories_str = ", ".join(existing_categories)

    prompt = f"""
    Analyze the following text and determine:
    1. The most appropriate broad category. If it doesn't fit into any existing categories ({categories_str}), suggest a new one.
    2. A main tag (1-3 words) that best describes the specific topic.
    3. Up to 4 additional tags that are relevant to the content.

    Text: {text}

    Format your response as follows:
    Category: [category name]
    Main Tag: [main tag]
    Additional Tags: [tag1], [tag2], [tag3], [tag4]
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )

    result = response.choices[0].message.content.strip().split("\n")
    category = result[0].split(": ")[1].strip()
    main_tag = result[1].split(": ")[1].strip()
    additional_tags = [tag.strip() for tag in result[2].split(": ")[1].split(",")]

    if category not in existing_categories:
        add_new_category(category)

    return category, main_tag, additional_tags


def format_idea(text):
    prompt = f"""
    Please format the following transcribed audio note, keeping it as natural as possible:
    1. Remove stuttering, repeated words, "uhms," and meaningless filler words.
    2. Preserve the original structure, flow of thoughts, and the speaker's unique voice and vocabulary.
    3. Use paragraph breaks to improve readability, especially for longer notes.
    4. Use bullet points only if the speaker naturally lists items.
    5. If the note contains a lot of information, use subtle formatting to improve clarity without changing the original structure drastically.
    6. Do not add summaries, main points, or change the content's meaning.

    Original transcription: {text}

    Format your response as a single, coherent note that feels natural and preserves the speaker's style.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


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


def clear_input_buffer():
    fd = sys.stdin.fileno()

    flags = fcntl.fcntl(fd, fcntl.F_GETFL)

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


def update_tag_index(tags, filepath):
    if os.path.exists(tag_index_file):
        with open(tag_index_file, "r") as f:
            tag_index = json.load(f)
    else:
        tag_index = {}

    for tag in tags:
        if tag in tag_index:
            tag_index[tag].append(filepath)
        else:
            tag_index[tag] = [filepath]

    with open(tag_index_file, "w") as f:
        json.dump(tag_index, f)


def process_audio(continue_last=False):
    global last_file
    audio = record_audio()

    print("Transcribing audio...")
    raw_text = transcribe_audio(audio)

    print("Formatting idea...")
    formatted_idea = format_idea(raw_text)

    if continue_last and last_file:
        category = os.path.dirname(last_file).split(os.path.sep)[-1]
        main_tag = os.path.basename(last_file).split("_")[1]
        additional_tags = []
    else:
        category, main_tag, additional_tags = determine_category_and_tags(raw_text)

    curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"note_{main_tag}_{curr_date}.txt"
    filepath = os.path.join(notes_dir, category, filename)

    if continue_last and last_file:
        with open(last_file, "a") as f:
            f.write(
                f"\n\nContinuation ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):\n\n"
            )
            f.write(formatted_idea)
        print(f"Continued note in {os.path.basename(last_file)}")
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(f"Category: {category}\n")
            f.write(f"Main Tag: {main_tag}\n")
            f.write(f"Additional Tags: {', '.join(additional_tags)}\n\n")
            f.write(formatted_idea)
        last_file = filepath
        print(f"New note saved as {filename} in category {category}")

    all_tags = [main_tag] + additional_tags
    update_tag_index(all_tags, filepath)


def transcribe_audio(audio):
    temp_file = "temp_audio.wav"
    wavfile.write(temp_file, 44100, audio)

    with open(temp_file, "rb") as file:
        transcription = client.audio.transcriptions.create(model="whisper-1", file=file)

    os.remove(temp_file)
    return transcription.text


def on_press(key):
    global should_stop
    if key == keyboard.Key.enter:
        should_stop = True
        return False  # Stop listener


def search_notes(query, category=None):
    if not os.path.exists(tag_index_file):
        print("No notes found.")
        return

    with open(tag_index_file, "r") as f:
        tag_index = json.load(f)

    if query.startswith("tag:"):
        tag = query[4:].strip().lower()
        if tag in tag_index:
            results = tag_index[tag]
        else:
            results = []
    else:
        results = []
        for filepath in set(
            [filepath for filepaths in tag_index.values() for filepath in filepaths]
        ):
            if category and category not in filepath:
                continue
            with open(filepath, "r") as f:
                content = f.read()
                if query.lower() in content.lower():
                    results.append(filepath)

    if results:
        print("Search results:")
        for filepath in results:
            print(f"- {os.path.relpath(filepath, notes_dir)}")
    else:
        print("No matching notes found.")


def display_categories():
    categories = get_existing_categories()
    print("Current categories:")
    for category in categories:
        print(f"- {category}")


def main():
    global last_file
    os.makedirs(notes_dir, exist_ok=True)

    while True:
        display_categories()
        clear_input_buffer()
        user_input = input(
            "Press Enter to start recording, 'C' to continue last note, 'S' to search, or 'Q' to quit: "
        ).lower()
        if user_input == "q":
            break
        elif user_input == "c":
            if last_file:
                process_audio(continue_last=True)
            else:
                print("No previous note to continue. Starting a new recording.")
                process_audio()
        elif user_input == "s":
            search_query = input(
                "Enter search terms (use 'tag:' prefix for tag search): "
            )
            category = input(
                "Enter category to search in (leave blank for all categories): "
            ).strip()
            search_notes(search_query, category if category else None)
        else:
            process_audio()
        print("Waiting for next command...")
        time.sleep(1)  # Add a small delay


if __name__ == "__main__":
    main()
