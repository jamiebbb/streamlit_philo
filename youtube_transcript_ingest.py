# import basics
import os
import json
import re
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# import supabase
from supabase.client import Client, create_client

# import tabulate for pretty printing
try:
    from tabulate import tabulate
except ImportError:
    print("Installing tabulate...")
    import subprocess
    subprocess.check_call(["pip", "install", "tabulate", "requests"])
    from tabulate import tabulate

# import csv handling
import csv
import os.path

# Define Supadata API constants
SUPADATA_API_URL = "https://api.supadata.ai"
SUPADATA_TRANSCRIPT_ENDPOINT = "/v1/youtube/transcript"
SUPADATA_API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiIsImtpZCI6IjEifQ.eyJpc3MiOiJuYWRsZXMiLCJpYXQiOiIxNzQ3OTIwNTIyIiwicHVycG9zZSI6ImFwaV9hdXRoZW50aWNhdGlvbiIsInN1YiI6IjEwMjk3YjAyYThlZjRhOTdhNmFjNjUwNjYxYWVlZjNiIn0.5OwI0aFR_BfgrDp2c55muHS9OyVX6XxHHPhULTzqdRY"

# Get the absolute path to the .env file
env_path = Path(__file__).parent / '.env'
print("Looking for .env file at:", env_path)

# load environment variables
load_dotenv(dotenv_path=env_path, override=True)  

# Debug prints to check environment variables
print("SUPABASE_URL:", os.environ.get("SUPABASE_URL"))
print("SUPABASE_SERVICE_KEY:", os.environ.get("SUPABASE_SERVICE_KEY")[:10] + "..." if os.environ.get("SUPABASE_SERVICE_KEY") else None)
print("OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY")[:10] + "..." if os.environ.get("OPENAI_API_KEY") else "NOT FOUND")

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize GPT-4o-mini model for metadata generation
metadata_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Extract YouTube video ID from URL
def extract_video_id(url):
    """Extract the video ID from a YouTube URL."""
    # Regular expressions for different YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard YouTube URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embedded URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # Shortened youtu.be URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

# Function to get video metadata from YouTube
def get_video_metadata(video_id):
    """Create metadata for a YouTube video by fetching title and channel info."""
    try:
        # Create basic metadata structure
        metadata = {
            "video_id": video_id,
            "source_url": f"https://www.youtube.com/watch?v={video_id}",
            "source_type": "youtube_video"
        }
        
        # Fetch the video page to extract title and channel info
        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            html_content = response.text
            
            # Extract title
            title_match = re.search(r'<meta name="title" content="([^"]+)"', html_content)
            if title_match:
                metadata["title"] = title_match.group(1)
            else:
                metadata["title"] = f"YouTube Video {video_id}"
                
            # Extract channel name (not the content author)
            channel_match = re.search(r'<link itemprop="name" content="([^"]+)"', html_content)
            if channel_match:
                metadata["youtube_channel"] = channel_match.group(1)
            else:
                # Try alternate pattern
                channel_match2 = re.search(r'"author":"([^"]+)"', html_content)
                if channel_match2:
                    metadata["youtube_channel"] = channel_match2.group(1)
                else:
                    metadata["youtube_channel"] = "Unknown Channel"
            
            print(f"Successfully extracted title: '{metadata['title']}' and channel: '{metadata['youtube_channel']}'")
        else:
            # Use placeholders if we can't fetch the page
            metadata["title"] = f"YouTube Video {video_id}"
            metadata["youtube_channel"] = "Unknown Channel"
            print(f"Couldn't fetch video page (HTTP {response.status_code}), using placeholder metadata")
        
        return metadata
    except Exception as e:
        print(f"Error creating video metadata: {e}")
        return {
            "title": f"YouTube Video {video_id}",
            "youtube_channel": "Unknown Channel",
            "video_id": video_id,
            "source_url": f"https://www.youtube.com/watch?v={video_id}",
            "source_type": "youtube_video"
        }

# Function to get transcript from Supadata API
def get_transcript(video_id, api_key=None):
    """Get transcript for a YouTube video using Supadata API."""
    try:
        # Use the global API key if none is provided
        if api_key is None:
            api_key = SUPADATA_API_KEY
            
        # Construct the API URL
        api_url = f"{SUPADATA_API_URL}{SUPADATA_TRANSCRIPT_ENDPOINT}"
        
        # Set up the request parameters
        params = {
            "videoId": video_id
        }
        
        # Set up headers with API key - use X-API-Key header which worked in testing
        headers = {
            "X-API-Key": api_key
        }
        
        print(f"Making API request for video: {video_id}")
        # Make the request to Supadata API
        response = requests.get(api_url, params=params, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract the transcript text from the response
            if "content" in data:
                # The transcript is in a "content" array with each segment having "text" fields
                # We need to join all the text segments together
                transcript = " ".join([segment.get("text", "") for segment in data.get("content", [])])
                return transcript
            else:
                print(f"Transcript content not found in API response for video {video_id}")
                return None
        else:
            print(f"Error calling Supadata API: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error getting transcript from Supadata API: {e}")
        return None

# Function to batch process multiple videos with Supadata API
def get_transcripts_batch(video_ids, api_key=None):
    """Get transcripts for multiple YouTube videos in a batch."""
    transcripts = {}
    
    try:
        # Use the global API key if none is provided
        if api_key is None:
            api_key = SUPADATA_API_KEY
            
        # Process videos in batches to avoid overloading the API
        batch_size = 5
        for i in range(0, len(video_ids), batch_size):
            batch = video_ids[i:i+batch_size]
            
            for video_id in batch:
                transcript = get_transcript(video_id, api_key)
                if transcript:
                    transcripts[video_id] = transcript
                
                # Be nice to the API with a small delay between requests
                time.sleep(1)
                
        return transcripts
    
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return transcripts

# Define audit function to check embeddings in Supabase (used optionally)
def audit_supabase_embeddings():
    print("\n========== CURRENT EMBEDDINGS IN SUPABASE ==========")
    try:
        # Get all documents from Supabase
        # Note: This will fetch all documents, which could be many
        result = supabase.table("documents").select("id, metadata").execute()
        documents = result.data
        
        if not documents:
            print("No documents found in Supabase.")
            return
        
        print(f"Total documents in database: {len(documents)}")
        
        # Count documents by title
        title_counts = {}
        metadata_by_title = {}
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            title = metadata.get("title", "Unknown")
            
            if title not in title_counts:
                title_counts[title] = 0
                metadata_by_title[title] = metadata
            
            title_counts[title] += 1
        
        # Display counts in a table
        table_data = []
        for title, count in title_counts.items():
            metadata = metadata_by_title[title]
            table_data.append([
                title, 
                count, 
                metadata.get("type", "Unknown"),
                metadata.get("source_type", "Unknown"),
                metadata.get("author", "Unknown"),
                metadata.get("difficulty", "Unknown"),
                metadata.get("tags", "Unknown")[:30] + "..." if metadata.get("tags", "Unknown") and len(str(metadata.get("tags", "Unknown"))) > 30 else metadata.get("tags", "Unknown")
            ])
        
        # Sort by number of chunks
        table_data.sort(key=lambda x: x[1], reverse=True)
        
        # Print table with pagination
        headers = ["Title", "Chunks", "Type", "Source", "Author", "Difficulty", "Tags"]
        
        # Ask user if they want to see all entries or just a summary
        print(f"\nFound {len(table_data)} unique documents.")
        print("How would you like to view the results?")
        print("1. All entries")
        print("2. First 10 entries")
        print("3. Summary only")
        
        choice = input("> ").strip()
        
        if choice == "1":
            # Show all entries
            print(tabulate(table_data, headers=headers, tablefmt="grid", maxcolwidths=[50, 10, 15, 15, 15, 15, 30]))
        elif choice == "2":
            # Show first 10 entries
            print(tabulate(table_data[:10], headers=headers, tablefmt="grid", maxcolwidths=[50, 10, 15, 15, 15, 15, 30]))
            print(f"... and {len(table_data)-10} more entries")
        else:
            # Just show summary stats
            print(f"Unique titles: {len(title_counts)}")
            print("Top 5 documents by chunk count:")
            print(tabulate(table_data[:5], headers=headers, tablefmt="grid", maxcolwidths=[50, 10, 15, 15, 15, 15, 30]))
        
    except Exception as e:
        print(f"Error during audit: {e}")
        import traceback
        traceback.print_exc()

# Define metadata generation function
def generate_metadata(title, transcript_sample, video_metadata):
    """Generate metadata for a YouTube transcript."""
    
    # Simple prompt with the title and transcript sample
    system_message = """You are a metadata expert who creates high-quality content summaries and tags for YouTube videos.
    Follow these instructions carefully:
    1. Create a concise summary using Orwell's writing rules (clear, concise language with active voice)
    2. Identify the genre/topic and content type - don't use "non-fiction" or "fiction" as a genre
    3. Identify the ACTUAL AUTHOR of the content (not the YouTube channel) from the title and content
    4. Assign a difficulty rating (beginner, intermediate, expert) based on complexity and target audience
    5. Generate relevant tags that would be useful in a chatbot context
    
    Format your response exactly as follows:
    Summary: [Your summary here]
    Genre: [Genre]
    Topic: [Topic]
    Type: [Content type - should be "video"]
    Author: [The actual author/speaker of the content, not the YouTube channel]
    Tags: [tag1, tag2, tag3, etc.]
    Difficulty: [beginner/intermediate/expert]"""
    
    # Using a simpler approach to interact with the model
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Generate metadata for YouTube video titled '{title}' with this transcript sample: {transcript_sample[:1500]}..."}
    ]
    
    # Generate metadata
    response = metadata_llm.invoke(messages)
    
    # Parse response into structured metadata
    metadata_dict = {}
    try:
        response_text = response.content
        # Extract fields using regex
        metadata_dict["summary"] = re.search(r"Summary: (.*?)(?:\n|$)", response_text, re.DOTALL).group(1).strip()
        metadata_dict["genre"] = re.search(r"Genre: (.*?)(?:\n|$)", response_text).group(1).strip()
        metadata_dict["topic"] = re.search(r"Topic: (.*?)(?:\n|$)", response_text).group(1).strip()
        metadata_dict["type"] = re.search(r"Type: (.*?)(?:\n|$)", response_text).group(1).strip()
        metadata_dict["author"] = re.search(r"Author: (.*?)(?:\n|$)", response_text).group(1).strip()
        metadata_dict["tags"] = re.search(r"Tags: (.*?)(?:\n|$)", response_text).group(1).strip()
        metadata_dict["difficulty"] = re.search(r"Difficulty: (.*?)(?:\n|$)", response_text).group(1).strip()
        
        # Add the video metadata
        for key, value in video_metadata.items():
            # Don't overwrite author with the YouTube channel name
            if key != "author":
                metadata_dict[key] = value
        
        # Make sure we keep the youtube_channel field separate from author
        metadata_dict["youtube_channel"] = video_metadata.get("youtube_channel", "Unknown Channel")
        
    except (AttributeError, Exception) as e:
        print(f"Error parsing metadata: {e}")
        print(f"Raw response: {response.content}")
        # Fallback metadata
        metadata_dict = {
            "summary": "Summary extraction failed",
            "genre": "Unknown",
            "topic": "Unknown",
            "type": "video",
            "author": "Unknown",
            "tags": "Unknown",
            "difficulty": "Unknown"
        }
        # Add the video metadata
        metadata_dict.update(video_metadata)
    
    print(f"Identified author: {metadata_dict.get('author', 'Unknown')}")
    return metadata_dict

# Function to allow user to edit metadata for a document
def edit_document_metadata(metadata):
    """Allow user to edit metadata for a document."""
    print("\n========== EDIT METADATA ==========")
    print("Enter new values or press Enter to keep current values.")
    
    # Make a copy of the metadata to edit
    edited_metadata = metadata.copy()
    
    # Fields that can be edited
    editable_fields = [
        "title", "author", "type", "genre", "topic", "difficulty", "tags", "summary"
    ]
    
    for field in editable_fields:
        current_value = edited_metadata.get(field, "")
        # For summary, only show a preview in the prompt
        display_value = current_value
        if field == "summary" and len(str(current_value)) > 100:
            display_value = str(current_value)[:100] + "..."
            
        print(f"\nCurrent {field}: {display_value}")
        new_value = input(f"New {field} (Enter to keep current): ")
        if new_value.strip():
            edited_metadata[field] = new_value
    
    return edited_metadata

# Function to print a summary of the documents processed
def print_document_summary(document_groups):
    print("\n========== DOCUMENTS PROCESSED ==========")
    table_data = []
    
    for video_id, info in document_groups.items():
        metadata = info["metadata"]
        chunks = info["chunks"]
        title = metadata.get("title", "Unknown")
        # Truncate tags if too long
        tags = metadata.get("tags", "Unknown")
        if len(str(tags)) > 30:
            tags = str(tags)[:27] + "..."
            
        table_data.append([
            title,
            len(chunks),
            metadata.get("type", "Unknown"),
            metadata.get("genre", "Unknown"),
            metadata.get("difficulty", "Unknown"),
            tags
        ])
    
    headers = ["Title", "Chunks", "Type", "Genre", "Difficulty", "Tags"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", maxcolwidths=[30, 10, 15, 15, 15, 30]))
    
    total_chunks = sum(len(info["chunks"]) for info in document_groups.values())
    print(f"\nTotal videos processed: {len(document_groups)}")
    print(f"Total chunks created: {total_chunks}")

# Function to process multiple YouTube videos
def process_youtube_videos(video_urls, api_key=None):
    """Process multiple YouTube video transcripts."""
    document_groups = {}
    
    # Use the global API key if none is provided
    if api_key is None:
        api_key = SUPADATA_API_KEY
    
    # Extract video IDs from URLs
    video_ids = []
    id_to_url = {}
    for url in video_urls:
        video_id = extract_video_id(url)
        if not video_id:
            print(f"Invalid YouTube URL: {url}")
            continue
        video_ids.append(video_id)
        id_to_url[video_id] = url
    
    if not video_ids:
        print("No valid YouTube URLs provided.")
        return document_groups
    
    # Get video metadata for all videos
    print(f"Fetching metadata for {len(video_ids)} videos...")
    video_metadata = {}
    for video_id in video_ids:
        metadata = get_video_metadata(video_id)
        video_metadata[video_id] = metadata
        print(f"Retrieved metadata for: {metadata.get('title', 'Unknown')}")
    
    # Get transcripts for all videos in batch
    print(f"Fetching transcripts from Supadata API...")
    transcripts = get_transcripts_batch(video_ids, api_key)
    
    # Process each video with metadata and transcript
    for video_id in video_ids:
        if video_id not in transcripts:
            print(f"No transcript available for video: {id_to_url[video_id]}")
            continue
            
        print(f"\nProcessing video: {id_to_url[video_id]} (ID: {video_id})")
        transcript = transcripts[video_id]
        metadata = video_metadata[video_id]
        
        print(f"Retrieved transcript ({len(transcript)} characters)")
        
        # Clean and format transcript
        cleaned_transcript = clean_transcript(transcript)
        
        # Generate enhanced metadata
        enhanced_metadata = generate_metadata(metadata.get("title", "Unknown"), cleaned_transcript, metadata)
        print("Generated enhanced metadata")
        
        # Create document
        doc = Document(page_content=cleaned_transcript, metadata=enhanced_metadata)
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
        chunks = text_splitter.split_documents([doc])
        
        print(f"Split transcript into {len(chunks)} chunks")
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks)
            })
        
        document_groups[video_id] = {
            "metadata": enhanced_metadata,
            "chunks": chunks,
            "transcript": cleaned_transcript
        }
    
    return document_groups

# Define function to clean and format transcript with GPT-4o-mini
def clean_transcript(transcript):
    """Clean and format a transcript using GPT-4o-mini."""
    print("Cleaning and formatting transcript with GPT-4o-mini...")
    
    # Create prompt for the model
    system_prompt = """You are an expert in grammar corrections and textual structuring.

Correct the classification of the provided text, adding commas, periods, question marks and other symbols necessary for natural and consistent reading. Do not change any words, just adjust the punctuation according to the grammatical rules and context.

Organize your content using markdown, structuring it with titles, subtitles, lists or other protected elements to clearly highlight the topics and information captured. Leave it in English and remember to always maintain the original formatting.

Textual organization should always be a priority according to the content of the text, as well as the appropriate title, which must make sense."""
    
    # Limit transcript length if needed to fit token limits
    max_content_length = 12000  # Characters, not tokens
    if len(transcript) > max_content_length:
        print(f"Transcript is too long ({len(transcript)} chars), trimming to ~{max_content_length} chars")
        transcript = transcript[:max_content_length] + "\n\n[Transcript truncated due to length]"
    
    # Using a simpler approach to interact with the model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is a YouTube transcript that needs cleaning and formatting:\n\n{transcript}"}
    ]
    
    try:
        # Generate cleaned transcript
        response = metadata_llm.invoke(messages)
        cleaned_transcript = response.content
        
        print(f"Transcript cleaned and formatted. Original length: {len(transcript)}, New length: {len(cleaned_transcript)}")
        return cleaned_transcript
    except Exception as e:
        print(f"Error cleaning transcript: {e}")
        return transcript  # Return original if there's an error

# Function to save transcripts and metadata to CSV
def save_to_csv(document_groups, filename="youtube_transcripts.csv"):
    """Save transcripts and metadata to a CSV file."""
    print(f"\nSaving transcripts and metadata to {filename}...")
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Define CSV headers
            fieldnames = ['video_id', 'title', 'youtube_channel', 'author', 'genre', 
                         'topic', 'difficulty', 'tags', 'summary', 'transcript']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write each document's data
            for video_id, info in document_groups.items():
                metadata = info["metadata"]
                transcript = info["transcript"]
                
                # Create a row with selected metadata and the full transcript
                row = {
                    'video_id': video_id,
                    'title': metadata.get('title', ''),
                    'youtube_channel': metadata.get('youtube_channel', ''),
                    'author': metadata.get('author', ''),
                    'genre': metadata.get('genre', ''),
                    'topic': metadata.get('topic', ''),
                    'difficulty': metadata.get('difficulty', ''),
                    'tags': metadata.get('tags', ''),
                    'summary': metadata.get('summary', ''),
                    'transcript': transcript
                }
                
                writer.writerow(row)
            
        print(f"Successfully saved {len(document_groups)} videos to {filename}")
        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        import traceback
        traceback.print_exc()
        return False

# Function to load transcripts and metadata from CSV
def load_from_csv(filename="youtube_transcripts.csv"):
    """Load transcripts and metadata from a CSV file."""
    if not os.path.exists(filename):
        print(f"CSV file {filename} not found")
        return {}
    
    print(f"\nLoading transcripts and metadata from {filename}...")
    document_groups = {}
    
    try:
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                video_id = row.get('video_id')
                if not video_id:
                    continue
                
                # Create metadata dictionary from CSV fields
                metadata = {
                    'title': row.get('title', ''),
                    'youtube_channel': row.get('youtube_channel', ''),
                    'author': row.get('author', ''),
                    'genre': row.get('genre', ''),
                    'topic': row.get('topic', ''),
                    'difficulty': row.get('difficulty', ''),
                    'tags': row.get('tags', ''),
                    'summary': row.get('summary', ''),
                    'video_id': video_id,
                    'source_url': f"https://www.youtube.com/watch?v={video_id}",
                    'source_type': 'youtube_video'
                }
                
                transcript = row.get('transcript', '')
                
                # Create document
                doc = Document(page_content=transcript, metadata=metadata)
                
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
                chunks = text_splitter.split_documents([doc])
                
                # Add chunk-specific metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    })
                
                document_groups[video_id] = {
                    "metadata": metadata,
                    "chunks": chunks,
                    "transcript": transcript
                }
        
        print(f"Successfully loaded {len(document_groups)} videos from {filename}")
        return document_groups
    except Exception as e:
        print(f"Error loading from CSV: {e}")
        import traceback
        traceback.print_exc()
        return {}

# Main function
def main():
    print("\n========== YOUTUBE TRANSCRIPT INGESTION ==========")
    
    # Set up API key
    api_key = SUPADATA_API_KEY
    print("Using configured Supadata API key.")
    print("You can change this key in the script if needed.")
    
    # Ask if user wants to load from CSV
    print("\nDo you want to load previously saved transcripts from CSV? (yes/no)")
    choice = input("> ").strip().lower()
    
    document_groups = {}
    if choice == "yes" or choice == "y":
        # Ask for filename
        print("Enter CSV filename (leave blank for default 'youtube_transcripts.csv'):")
        filename = input("> ").strip()
        if not filename:
            filename = "youtube_transcripts.csv"
            
        # Load from CSV
        document_groups = load_from_csv(filename)
        
        if not document_groups:
            print("No documents loaded from CSV or CSV file not found.")
            
    # If no documents loaded from CSV or user chose not to load, proceed with regular flow
    if not document_groups:
        # Ask for YouTube URLs
        print("\nEnter YouTube URLs (one per line, blank line to finish):")
        urls = []
        while True:
            url = input("> ").strip()
            if not url:
                break
            urls.append(url)
        
        if not urls:
            print("No URLs provided. Exiting.")
            return
        
        print(f"\nProcessing {len(urls)} YouTube videos...")
        
        # Process videos
        try:
            document_groups = process_youtube_videos(video_urls=urls, api_key=api_key)
            
            if not document_groups:
                print("No documents were processed successfully.")
                return
                
            # Ask if user wants to save to CSV
            print("\nDo you want to save these transcripts to CSV for future use? (yes/no)")
            save_choice = input("> ").strip().lower()
            if save_choice == "yes" or save_choice == "y":
                # Ask for filename
                print("Enter CSV filename (leave blank for default 'youtube_transcripts.csv'):")
                filename = input("> ").strip()
                if not filename:
                    filename = "youtube_transcripts.csv"
                    
                # Save to CSV
                save_to_csv(document_groups, filename)
            
        except Exception as e:
            print(f"Error processing videos: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Continue with normal flow - print summary, allow editing, etc.
    # Print summary
    print_document_summary(document_groups)
    
    # Allow editing metadata
    while True:
        print("\nWould you like to edit metadata for any video? (yes/no)")
        choice = input("> ").strip().lower()
        if choice != "yes" and choice != "y":
            break
            
        # List documents with numbers for selection
        print("\nSelect a video to edit:")
        video_ids = list(document_groups.keys())
        for i, video_id in enumerate(video_ids):
            title = document_groups[video_id]["metadata"].get("title", "Unknown")
            print(f"{i+1}. {title} ({video_id})")
            
        # Get document selection
        selection = input("Enter video number (or 0 to cancel): ")
        try:
            selection = int(selection)
            if selection == 0:
                continue
            if 1 <= selection <= len(video_ids):
                selected_id = video_ids[selection-1]
                video_info = document_groups[selected_id]
                
                # Edit metadata
                edited_metadata = edit_document_metadata(video_info["metadata"])
                
                # Apply edited metadata to all chunks
                video_info["metadata"] = edited_metadata
                for chunk in video_info["chunks"]:
                    chunk.metadata = edited_metadata.copy()
                    # Restore chunk-specific metadata
                    chunk.metadata["chunk_id"] = chunk.metadata.get("chunk_id", 0)
                    chunk.metadata["total_chunks"] = chunk.metadata.get("total_chunks", len(video_info["chunks"]))
                    
                print("Metadata updated.")
                
                # Ask if user wants to save updated metadata to CSV
                print("\nDo you want to save the updated metadata to CSV? (yes/no)")
                save_choice = input("> ").strip().lower()
                if save_choice == "yes" or save_choice == "y":
                    # Ask for filename
                    print("Enter CSV filename (leave blank for default 'youtube_transcripts.csv'):")
                    filename = input("> ").strip()
                    if not filename:
                        filename = "youtube_transcripts.csv"
                        
                    # Save to CSV
                    save_to_csv(document_groups, filename)
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a number.")
    
    # Ask for confirmation before uploading to Supabase
    print("\nDo you want to see what's currently in Supabase? (yes/no)")
    choice = input("> ").strip().lower()
    if choice == "yes" or choice == "y":
        audit_supabase_embeddings()
    
    # Ask for confirmation before uploading to Supabase
    print("\nDo you want to upload these videos to Supabase? (yes/no)")
    choice = input("> ").strip().lower()
    if choice != "yes" and choice != "y":
        print("Upload cancelled.")
        return
    
    # Combine all chunks for uploading
    all_chunks = []
    for video_id, info in document_groups.items():
        all_chunks.extend(info["chunks"])
    
    # Store chunks in vector store
    print(f"Storing {len(all_chunks)} chunks in vector store...")
    start_time = time.time()
    vector_store = SupabaseVectorStore.from_documents(
        all_chunks,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
        chunk_size=1000,
    )
    end_time = time.time()
    
    print(f"Ingestion complete! Uploaded {len(all_chunks)} chunks in {end_time - start_time:.2f} seconds")
    
    # Ask if user wants to see final state in Supabase
    print("\nDo you want to see the updated content in Supabase? (yes/no)")
    choice = input("> ").strip().lower()
    if choice == "yes" or choice == "y":
        audit_supabase_embeddings()

# Run the main function if script is executed directly
if __name__ == "__main__":
    main() 