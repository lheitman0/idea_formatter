import openai
import sounddevice as sd
import numpy as np
import wave
import os
from datetime import datetime

# Initialize OpenAI client (you'll need to set your API key as an environment variable)
client = openai.OpenAI()

def record_audio(duration=10, sample_rate=44100):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Recording finished")
    return audio

def save_audio(audio, filename, sample_rate=44100):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

def transcribe_audio(audio_file):
    with open(audio_file, "rb") as file:
        transcription = client.audio.transcriptions.create(model="whisper-1", file=file)
    return transcription.text

def format_idea(text):
    prompt = f"""
    Format the following idea into a structured format:
    1. Main Idea (1 sentence summary)
    2. Key Points (bullet points)
    3. Additional Notes (if any)

    Idea: {text}
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def process_audio():
    # Create a new directory for this session
    session_dir = "audio_session"
    os.makedirs(session_dir, exist_ok=True)

    # Record audio
    audio = record_audio()

    # Save audio file
    audio_file = os.path.join(session_dir, "recorded_audio.wav")
    save_audio(audio, audio_file)

    # Transcribe audio
    raw_text = transcribe_audio(audio_file)
    print("\nRaw text:")
    print(raw_text)

    # Format idea
    formatted_idea = format_idea(raw_text)
    print("\nFormatted idea:")
    print(formatted_idea)

    # Save to file
    with open(os.path.join(session_dir, f"idea_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"), "w") as f:
        f.write(f"Raw text:\n{raw_text}\n\nFormatted idea:\n{formatted_idea}")

    print(f"\nIdea saved in {session_dir}/idea.txt")

def main():
    while True:
        user_input = input("Press Enter to start recording an idea (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        process_audio()

if __name__ == "__main__":
    main()