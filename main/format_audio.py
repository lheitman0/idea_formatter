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
from scipy.io import wavfile
import json

client = openai.OpenAI()

last_file = None
session_dir = "audio_sessions"
tag_index_file = "tag_index.json"

def get_existing_categories():
    return [d for d in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, d))]

def add_new_category(category):
    category_path = os.path.join(session_dir, category)
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

def update_tag_index(tags, filepath):
    if os.path.exists(tag_index_file):
        with open(tag_index_file, 'r') as f:
            tag_index = json.load(f)
    else:
        tag_index = {}
    
    for tag in tags:
        if tag in tag_index:
            tag_index[tag].append(filepath)
        else:
            tag_index[tag] = [filepath]
    
    with open(tag_index_file, 'w') as f:
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
        main_tag = os.path.basename(last_file).split('_')[1]
        additional_tags = []
    else:
        category, main_tag, additional_tags = determine_category_and_tags(raw_text)

    curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"note_{main_tag}_{curr_date}.txt"
    filepath = os.path.join(session_dir, category, filename)

    if continue_last and last_file:
        with open(last_file, "a") as f:
            f.write(f"\nContinuation ({curr_date}):\n\n")
            f.write(f"Raw Transcription:\n{raw_text}\n\n")
            f.write(f"Formatted Idea:\n{formatted_idea}\n")
        print(f"Continued note in {os.path.basename(last_file)}")
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(f"Category: {category}\n")
            f.write(f"Main Tag: {main_tag}\n")
            f.write(f"Additional Tags: {', '.join(additional_tags)}\n\n")
            f.write(f"Raw Transcription:\n{raw_text}\n\n")
            f.write(f"Formatted Idea:\n{formatted_idea}\n")
        last_file = filepath
        print(f"New note saved as {filename} in category {category}")

    all_tags = [main_tag] + additional_tags
    update_tag_index(all_tags, filepath)

def search_notes(query, category=None):
    if not os.path.exists(tag_index_file):
        print("No notes found.")
        return

    with open(tag_index_file, 'r') as f:
        tag_index = json.load(f)

    if query.startswith("tag:"):
        tag = query[4:].strip().lower()
        if tag in tag_index:
            results = tag_index[tag]
        else:
            results = []
    else:
        results = []
        for filepath in set([filepath for filepaths in tag_index.values() for filepath in filepaths]):
            if category and category not in filepath:
                continue
            with open(filepath, 'r') as f:
                content = f.read()
                if query.lower() in content.lower():
                    results.append(filepath)
    
    if results:
        print("Search results:")
        for filepath in results:
            print(f"- {os.path.relpath(filepath, session_dir)}")
    else:
        print("No matching notes found.")

def display_categories():
    categories = get_existing_categories()
    print("Current categories:")
    for category in categories:
        print(f"- {category}")

def main():
    global last_file
    os.makedirs(session_dir, exist_ok=True)
    
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
            search_query = input("Enter search terms (use 'tag:' prefix for tag search): ")
            category = input("Enter category to search in (leave blank for all categories): ").strip()
            search_notes(search_query, category if category else None)
        else:
            process_audio()
        print("Waiting for next command...")
        time.sleep(1)  # Add a small delay

if __name__ == "__main__":
    main()