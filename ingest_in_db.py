# import basics
import os
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# import supabase
from supabase.client import Client, create_client

# import tabulate for pretty printing
try:
    from tabulate import tabulate
except ImportError:
    print("Installing tabulate for better display...")
    import subprocess
    subprocess.check_call(["pip", "install", "tabulate"])
    from tabulate import tabulate

# Get the absolute path to the .env file
env_path = Path(__file__).parent / '.env'
print("Looking for .env file at:", env_path)

# load environment variables
load_dotenv(dotenv_path=env_path)  

# Force the correct OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Debug prints to check environment variables
print("SUPABASE_URL:", os.environ.get("SUPABASE_URL"))
print("SUPABASE_SERVICE_KEY:", os.environ.get("SUPABASE_SERVICE_KEY")[:10] + "..." if os.environ.get("SUPABASE_SERVICE_KEY") else None)
print("OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize GPT-4o-mini model for metadata generation
metadata_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define audit function to check embeddings in Supabase (used optionally)
def audit_supabase_embeddings():
    print("\n========== CURRENT EMBEDDINGS IN SUPABASE ==========")
    try:
        # Get all documents from Supabase
        result = supabase.table("documents").select("id, metadata").execute()
        documents = result.data
        
        if not documents:
            print("No documents found in Supabase.")
            return
        
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
                metadata.get("genre", "Unknown"),
                metadata.get("difficulty", "Unknown"),
                metadata.get("tags", "Unknown")
            ])
        
        # Sort by number of chunks
        table_data.sort(key=lambda x: x[1], reverse=True)
        
        # Print table
        headers = ["Title", "Chunks", "Type", "Genre", "Difficulty", "Tags"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print(f"\nTotal documents in database: {len(documents)}")
        print(f"Unique titles: {len(title_counts)}")
        
    except Exception as e:
        print(f"Error during audit: {e}")

# Define metadata generation function
def generate_metadata(content, doc_metadata):
    # Extract title and author from filename or existing metadata
    title = doc_metadata.get("source", "").split("/")[-1]
    # Remove file extension if present
    title = re.sub(r"\.\w+$", "", title)
    author = doc_metadata.get("author", "Unknown Author")
    
    # Simple prompt with just the title - since that's sufficient
    system_message = """You are a metadata expert who creates high-quality content summaries and tags.
    Follow these instructions carefully:
    1. Create a concise summary using Orwell's writing rules (clear, concise language with active voice)
    2. Identify the genre/topic and content type (book, video, podcast, research paper, etc.) - don't use "non-fiction" or "fiction" as a genre
    3. Assign a difficulty rating (beginner, intermediate, expert) based on complexity and target audience
    4. Generate relevant tags that would be useful in a chatbot context
    
    Format your response exactly as follows:
    Summary: [Your summary here]
    Genre: [Genre]
    Topic: [Topic]
    Type: [Content type]
    Tags: [tag1, tag2, tag3, etc.]
    Difficulty: [beginner/intermediate/expert]"""
    
    # Using a simpler approach to interact with the model
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Generate metadata for: {title}"}
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
        metadata_dict["tags"] = re.search(r"Tags: (.*?)(?:\n|$)", response_text).group(1).strip()
        metadata_dict["difficulty"] = re.search(r"Difficulty: (.*?)(?:\n|$)", response_text).group(1).strip()
        metadata_dict["title"] = title
        metadata_dict["author"] = author
    except (AttributeError, Exception) as e:
        print(f"Error parsing metadata: {e}")
        print(f"Raw response: {response.content}")
        # Fallback metadata
        metadata_dict = {
            "summary": "Summary extraction failed",
            "genre": "Unknown",
            "topic": "Unknown",
            "type": "Unknown",
            "tags": "Unknown",
            "difficulty": "Unknown",
            "title": title,
            "author": author
        }
    
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
        if field == "summary" and len(current_value) > 100:
            display_value = current_value[:100] + "..."
            
        print(f"\nCurrent {field}: {display_value}")
        new_value = input(f"New {field} (Enter to keep current): ")
        if new_value.strip():
            edited_metadata[field] = new_value
    
    return edited_metadata

# Function to print a summary of the documents processed
def print_document_summary(doc_groups):
    print("\n========== DOCUMENTS PROCESSED ==========")
    table_data = []
    
    for source, chunks in doc_groups.items():
        metadata = chunks[0].metadata
        title = metadata.get("title", "Unknown")
        # Truncate tags if too long
        tags = metadata.get("tags", "Unknown")
        if len(tags) > 30:
            tags = tags[:27] + "..."
            
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
    
    total_chunks = sum(len(chunks) for chunks in doc_groups.values())
    print(f"\nTotal documents processed: {len(doc_groups)}")
    print(f"Total chunks created: {total_chunks}")

# Function to list files in documents folder
def list_documents_folder():
    documents_dir = Path("documents")
    if not documents_dir.exists():
        print("The 'documents' folder does not exist.")
        return
    
    print("\n========== FILES IN DOCUMENTS FOLDER ==========")
    files = list(documents_dir.glob("**/*.*"))
    
    if not files:
        print("No files found in the documents folder.")
        return
    
    table_data = []
    for file in files:
        try:
            # Get file size in KB
            size_kb = file.stat().st_size / 1024
            # Get file modification time
            mod_time = time.ctime(file.stat().st_mtime)
            # Get file extension
            ext = file.suffix.lower()
            
            # Truncate very long filenames for display
            display_name = file.name
            if len(display_name) > 80:
                display_name = display_name[:77] + "..."
            
            table_data.append([
                display_name,
                f"{size_kb:.1f} KB",
                mod_time,
                ext
            ])
        except (OSError, FileNotFoundError, PermissionError) as e:
            # Handle files that can't be accessed due to long paths or permissions
            print(f"Warning: Could not access file '{file.name}': {e}")
            table_data.append([
                file.name[:77] + "..." if len(file.name) > 80 else file.name,
                "Unknown",
                "Unknown",
                file.suffix.lower() if file.suffix else "Unknown"
            ])
    
    headers = ["Filename", "Size", "Last Modified", "Type"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\nTotal files: {len(files)}")

# Main function
def main():
    print("\n========== DOCUMENT INGESTION PROCESS ==========")
    
    # First, list files in the documents folder
    list_documents_folder()
    
    # load pdf docs from folder 'documents'
    print("\nLoading documents from 'documents' folder...")
    loader = PyPDFDirectoryLoader("documents")
    
    # split the documents in multiple chunks
    print("Splitting documents into chunks...")
    try:
        documents = loader.load()
        if not documents:
            print("No documents loaded. Please check the 'documents' folder.")
            return
            
        print(f"Loaded {len(documents)} document(s).")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        print(f"Split into {len(docs)} chunks.")
        
        # Process documents to add metadata
        print(f"Generating metadata...")
        # Group chunks by source document
        doc_groups = {}
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            if source not in doc_groups:
                doc_groups[source] = []
            doc_groups[source].append(doc)
        
        # Generate metadata for each source document
        for source, chunks in doc_groups.items():
            print(f"Processing document: {source}")
            # Combine content from first few chunks to get a representative sample
            sample_content = " ".join([chunk.page_content for chunk in chunks[:3]])
            # Generate metadata using the combined content
            metadata = generate_metadata(sample_content, chunks[0].metadata)
            
            # Add metadata to all chunks from this document
            for chunk in chunks:
                chunk.metadata.update(metadata)
                # Add chunk-specific metadata
                chunk.metadata["chunk_id"] = chunks.index(chunk)
                chunk.metadata["total_chunks"] = len(chunks)
        
        print(f"Enhanced {len(docs)} chunks with metadata")
        
        # Debug: Show final chunk content for each document
        print("\n========== DEBUG: FINAL CHUNK CONTENT ==========")
        for source, chunks in doc_groups.items():
            title = chunks[0].metadata.get("title", "Unknown")
            final_chunk = chunks[-1]  # Get the last chunk
            
            print(f"\nDocument: {title}")
            print(f"Total chunks: {len(chunks)}")
            print(f"Final chunk ({len(chunks)}/{len(chunks)}) content preview:")
            print("-" * 80)
            
            # Show first 500 characters of final chunk
            final_content = final_chunk.page_content
            if len(final_content) > 500:
                print(final_content[:500] + "...")
                print(f"\n[Final chunk contains {len(final_content)} characters total]")
            else:
                print(final_content)
                print(f"\n[Final chunk contains {len(final_content)} characters total]")
            
            print("-" * 80)
            
            # Show chunk metadata
            print(f"Final chunk metadata:")
            print(f"  - Chunk ID: {final_chunk.metadata.get('chunk_id', 'Unknown')}")
            print(f"  - Total chunks: {final_chunk.metadata.get('total_chunks', 'Unknown')}")
            print(f"  - Source: {final_chunk.metadata.get('source', 'Unknown')}")
        
        # Ask if user wants to see more detailed debug info
        print("\nWould you like to see detailed chunk breakdown for any document? (yes/no)")
        debug_choice = input("> ").strip().lower()
        if debug_choice == "yes" or debug_choice == "y":
            # List documents for detailed inspection
            print("\nSelect a document for detailed chunk analysis:")
            source_list = list(doc_groups.keys())
            for i, source in enumerate(source_list):
                title = doc_groups[source][0].metadata.get("title", "Unknown")
                print(f"{i+1}. {title} ({len(doc_groups[source])} chunks)")
                
            selection = input("Enter document number (or 0 to skip): ")
            try:
                selection = int(selection)
                if 1 <= selection <= len(source_list):
                    selected_source = source_list[selection-1]
                    chunks = doc_groups[selected_source]
                    
                    print(f"\n========== DETAILED CHUNK ANALYSIS ==========")
                    print(f"Document: {chunks[0].metadata.get('title', 'Unknown')}")
                    print(f"Total chunks: {len(chunks)}")
                    
                    for i, chunk in enumerate(chunks):
                        print(f"\n--- Chunk {i+1}/{len(chunks)} ---")
                        content_preview = chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
                        print(f"Content preview: {content_preview}")
                        print(f"Character count: {len(chunk.page_content)}")
                        
                        # Show first and last few words to verify continuity
                        words = chunk.page_content.split()
                        if len(words) > 10:
                            print(f"First 5 words: {' '.join(words[:5])}")
                            print(f"Last 5 words: {' '.join(words[-5:])}")
                        
                        if i < len(chunks) - 1:  # Not the last chunk
                            print("(Press Enter to continue to next chunk, or 'q' to quit)")
                            user_input = input()
                            if user_input.lower() == 'q':
                                break
            except ValueError:
                print("Invalid selection, skipping detailed analysis.")
        
        # Print summary of documents processed
        print_document_summary(doc_groups)
        
        # Add document coverage summary
        print("\n========== DOCUMENT COVERAGE SUMMARY ==========")
        total_chars_all_docs = 0
        for source, chunks in doc_groups.items():
            title = chunks[0].metadata.get("title", "Unknown")
            total_chars_this_doc = sum(len(chunk.page_content) for chunk in chunks)
            total_chars_all_docs += total_chars_this_doc
            
            # Calculate average chunk size
            avg_chunk_size = total_chars_this_doc / len(chunks) if chunks else 0
            
            print(f"\n📄 {title}:")
            print(f"   • Total chunks: {len(chunks)}")
            print(f"   • Total characters: {total_chars_this_doc:,}")
            print(f"   • Average chunk size: {avg_chunk_size:.0f} characters")
            print(f"   • First chunk starts with: '{chunks[0].page_content[:50]}...'")
            print(f"   • Last chunk ends with: '...{chunks[-1].page_content[-50:]}'")
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   • Total documents: {len(doc_groups)}")
        print(f"   • Total chunks: {len(docs)}")
        print(f"   • Total characters across all documents: {total_chars_all_docs:,}")
        print(f"   • Average characters per document: {total_chars_all_docs / len(doc_groups):,.0f}")
        print(f"   • Average chunks per document: {len(docs) / len(doc_groups):.1f}")
        
        # Verify chunk continuity
        print(f"\n🔍 CHUNK CONTINUITY CHECK:")
        for source, chunks in doc_groups.items():
            title = chunks[0].metadata.get("title", "Unknown")
            print(f"   • {title}: Chunks 1-{len(chunks)} ✅")
        
        # Allow editing metadata
        while True:
            print("\nWould you like to edit metadata for any document? (yes/no)")
            choice = input("> ").strip().lower()
            if choice != "yes" and choice != "y":
                break
                
            # List documents with numbers for selection
            print("\nSelect a document to edit:")
            source_list = list(doc_groups.keys())
            for i, source in enumerate(source_list):
                title = doc_groups[source][0].metadata.get("title", "Unknown")
                print(f"{i+1}. {title} ({source})")
                
            # Get document selection
            selection = input("Enter document number (or 0 to cancel): ")
            try:
                selection = int(selection)
                if selection == 0:
                    continue
                if 1 <= selection <= len(source_list):
                    selected_source = source_list[selection-1]
                    chunks = doc_groups[selected_source]
                    
                    # Edit metadata
                    edited_metadata = edit_document_metadata(chunks[0].metadata)
                    
                    # Apply edited metadata to all chunks
                    for chunk in chunks:
                        chunk.metadata.update(edited_metadata)
                        
                    print("Metadata updated.")
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
        print("\nDo you want to upload these documents to Supabase? (yes/no)")
        choice = input("> ").strip().lower()
        if choice != "yes" and choice != "y":
            print("Upload cancelled.")
            return
        
        # store chunks in vector store
        print("Storing chunks in vector store...")
        start_time = time.time()
        vector_store = SupabaseVectorStore.from_documents(
            docs,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=1000,
        )
        end_time = time.time()
        
        print(f"Ingestion complete! Uploaded {len(docs)} chunks in {end_time - start_time:.2f} seconds")
        
        # Ask if user wants to see final state in Supabase
        print("\nDo you want to see the updated content in Supabase? (yes/no)")
        choice = input("> ").strip().lower()
        if choice == "yes" or choice == "y":
            audit_supabase_embeddings()
            
    except Exception as e:
        print(f"Error processing documents: {e}")
        import traceback
        traceback.print_exc()

# Run the main function if script is executed directly
if __name__ == "__main__":
    main()