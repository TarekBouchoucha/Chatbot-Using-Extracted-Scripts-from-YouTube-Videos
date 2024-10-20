import os
import argparse
import csv
import requests
import scrapetube
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
from subprocess import run
import chromadb
from chromadb.utils import embedding_functions

import yt_dlp
import openai


# Set paramaters for audio extraction later
AUDIO_FORMAT = "mp3"
PREFERRED_QUALITY = "96"
MAX_FILESIZE = 25 * 1024 * 1024  # 25MB
FFMPEG_AUDIO_CHANNELS = "1"  # Mono
FFMPEG_BITRATE = "32k"

# Initialize clients
Groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

openai_client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

# Utility function to split text into chunks
def split_text(text, chunk_size=512):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to get channel ID from the channel name
def get_channel_id(channel_name):
    url = f"https://youtube.googleapis.com/youtube/v3/search?part=snippet&q={channel_name}&type=channel&key={os.environ.get("Youtube_Data_API_Key")}"
    headers = {'Accept': 'application/json'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if 'items' in data and len(data['items']) > 0:
            return data['items'][0]['snippet']['channelId']
        else:
            print("No channels found")
            return None
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None

# Function to get video transcripts
def get_video_transcript(video_id):
    try:
        video_data = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([x['text'] for x in video_data])
    except Exception as e:
        print(f"An error occurred fetching the transcript for video {video_id}: {e}")
        return None

# Function to summarize text using the Groq API
def summarize_text(text_chunks):
    summarized_chunks = []
    for chunk in text_chunks:
        response = Groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize this in the form of facts, "
                        "keeping numerical values for precision. "
                        "Provide the summary as facts without "
                        f"mentioning it's from a video: \n{chunk}"
                    )
                }
            ],
            model="llama3-8b-8192",
        )
        summary = response.choices[0].message.content
        summarized_chunks.append(summary)
        
    return ' '.join(summarized_chunks)

def download_audio_from_youtube(url):
    """Downloads audio from the given YouTube URL and returns the filename."""

    filename = None

    def my_hook(d):
        nonlocal filename
        if d["status"] == "finished":
            filename = d["filename"]

    ydl_opts = { 
        "outtmpl": "%(title)s.%(ext)s",
        "format": "worstaudio",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": AUDIO_FORMAT,
                "preferredquality": PREFERRED_QUALITY,
            }
        ],
        "max_filesize": MAX_FILESIZE,
        "progress_hooks": [my_hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Strip the extension from the filename to use it for further processing
    if filename:
        return filename.rsplit(".", 1)[0]

def convert_audio_to_mono(audio_filename):
    """Converts the downloaded audio file to mono format with lower bitrate."""
    command = [
        "ffmpeg",
        "-i",
        f"{audio_filename}.{AUDIO_FORMAT}",
        "-ac",
        FFMPEG_AUDIO_CHANNELS,
        "-ab",
        FFMPEG_BITRATE,
        "-y",
        f"{audio_filename}_mono.{AUDIO_FORMAT}",
    ]
    run(command)

def transcribe_audio(audio_filename):
    with open(f"{audio_filename}_mono.{AUDIO_FORMAT}", "rb") as audio_file:
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-large-v3", file=audio_file, response_format="text"
        )
    return transcription

def transcribe_from_audio(url):
    try:
        audio_filename = download_audio_from_youtube(url)
        convert_audio_to_mono(audio_filename)
        transcript = transcribe_audio(audio_filename)
        return transcript
    finally:
        # Cleanup downloaded files only if they exist
        if os.path.exists(f"{audio_filename}.{AUDIO_FORMAT}"):
            os.remove(f"{audio_filename}.{AUDIO_FORMAT}")
        if os.path.exists(f"{audio_filename}_mono.{AUDIO_FORMAT}"):
            os.remove(f"{audio_filename}_mono.{AUDIO_FORMAT}")

# Function to process and save video data to a CSV file
def process_videos_to_csv(channel_id, csv_file, chunk_size=8000):
    videos = scrapetube.get_channel(channel_id)
    
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['URL', 'Title', 'Transcript', 'Summary'])
        
        for video in videos:
            video_id = str(video['videoId'])
            url = f"https://www.youtube.com/watch?v={video_id}"
            title = video['title']['runs'][0]['text']
            
            try:
                print(f'Processing video: {title}')
                try:
                    transcript = get_video_transcript(video_id)
                    text_chunks = split_text(transcript, chunk_size=chunk_size)
                    final_summary = summarize_text(text_chunks)
                    writer.writerow([url, title, transcript, final_summary])
                except Exception as e:
                    #if the transcript of the video isn't available (which is very unlikely)
                    #try with the another methode by transcribing from the audio
                    transcript=transcribe_from_audio(url)
                    text_chunks = split_text(transcript, chunk_size=chunk_size)
                    final_summary = summarize_text(text_chunks)
                    writer.writerow([url, title, transcript, final_summary])
            except Exception as e:
                print(f"An error occurred with video {url}: {e}")
                continue

def store_in_csv(channel_name, csv_file, chunk_size):
    channel_id = get_channel_id(channel_name)
    
    if channel_id:
        print(f'Processing channel: {channel_name} with ID: {channel_id}')
        process_videos_to_csv(channel_id, csv_file, chunk_size)
    else:
        print("Failed to retrieve channel ID. Exiting...")

#store the csv data in a vector database
def store_in_vector_database(file_path, database_path, collection_name, embedding_function_model):
    """
    Creates a ChromaDB collection from a CSV file containing documents and metadata.

    Args:
        file_path (str): The path to the CSV file.
        database_path(str): The path to the database
        collection_name (str): The name of the ChromaDB collection.
        embedding_function_model (str): The name of the SentenceTransformer model for embeddings.

    Returns:
        chromadb.Client: The ChromaDB client object.
    """

    with open(file_path, encoding='utf-8') as file:
        lines = csv.reader(file)

        documents = []
        metadatas = []
        ids = []
        id = 1

        for i, line in enumerate(lines):
            if i == 0:
                continue

            documents.append(line[3])
            metadatas.append({"item_id": line[0]})
            ids.append(str(id))
            id += 1

    chroma_client = chromadb.PersistentClient(path=database_path)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_function_model)
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=sentence_transformer_ef)
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    return chroma_client


# Main function to to parse arguments and orchestrate the process
def main(): 
    parser = argparse.ArgumentParser(description="Summarize YouTube videos from channel url and sote them in a csv.")
    parser.add_argument(
        "url", type=str, help="The URL of the YouTube channel."
    )
    args = parser.parse_args()
    channel_url = args.url.replace("\\", "")
    channel_name = channel_url.replace('https://www.youtube.com/@', '')
    csv_file = f'{channel_name}.csv'
    chunk_size = 8000
    database_path="My_database"
    collection_name = "Agriculture"
    embedding_function_model = "all-mpnet-base-v2"
    
    store_in_csv(channel_name, csv_file, chunk_size)

    print("\n######### finished storing data in csv #########\n")

    store_in_vector_database(csv_file, database_path, collection_name, embedding_function_model)


# Entry point
if __name__ == "__main__":
    main()