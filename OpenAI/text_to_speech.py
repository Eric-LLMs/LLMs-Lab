import openai
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

# Set OpenAI API Key
openai.api_key = api_key


# Read content from a TXT file
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# Call OpenAI's TTS API and save as MP3
def text_to_speech(text, filename):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",  # Optional: "alloy", "echo", "fable", "onyx", "nova", "shimmer"
        input=text
    )
    with open(filename, "wb") as f:
        f.write(response.content)


# Process all chapter TXT files
def process_text_to_audio(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = sorted(os.listdir(input_folder))  # Ensure processing in chapter order
    for i, file in enumerate(files):
        if file.endswith(".txt"):
            file_path = os.path.join(input_folder, file)
            chapter_name = os.path.splitext(file)[0]
            output_file = os.path.join(output_folder, f"{chapter_name}.mp3")

            print(f"🔊 Generating speech: {chapter_name}...")
            text = read_text_file(file_path)
            text_to_speech(text, output_file)
            time.sleep(1)  # Avoid API rate limits

    print("🎧 All chapter speeches generated!")


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    text_folder = os.path.join(root_dir, "text_output")  # Directory for chapter TXT files
    print("Current Python file directory:", text_folder)

    # Run speech conversion program
    audio_folder = os.path.join(root_dir, "audio_output")  # Directory for MP3 files
    process_text_to_audio(text_folder, audio_folder)
