"""
Document Tracker for Agentic RAG System
Manages a CSV file to track all uploaded documents and prevent duplicates.
"""

import os
import pandas as pd
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import streamlit as st


class DocumentTracker:
    """
    Manages document tracking to prevent duplicates and maintain upload history.
    """
    
    def __init__(self, csv_file: str = "uploaded_documents_tracker.csv"):
        self.csv_file = csv_file
        self.ensure_csv_exists()
    
    def ensure_csv_exists(self):
        """Create the CSV file with proper headers if it doesn't exist."""
        if not os.path.exists(self.csv_file):
            # Create empty DataFrame with all required columns
            df = pd.DataFrame(columns=[
                "id",
                "title", 
                "author",
                "summary",
                "type",
                "genre", 
                "topic",
                "difficulty",
                "source_type",
                "tags",
                "chunks",
                "chunk_size",
                "chunk_overlap",
                "file_hash",
                "file_name",
                "file_size",
                "upload_date",
                "source_url",
                "video_id"
            ])
            df.to_csv(self.csv_file, index=False)
            print(f"✅ Created new document tracker CSV: {self.csv_file}")
    
    def calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA-256 hash of file content for duplicate detection."""
        return hashlib.sha256(file_content).hexdigest()
    
    def calculate_url_hash(self, url: str) -> str:
        """Calculate hash of URL for YouTube video duplicate detection."""
        return hashlib.sha256(url.encode()).hexdigest()
    
    def is_duplicate_file(self, file_content: bytes, file_name: str) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a file is a duplicate based on content hash.
        
        Returns:
            Tuple of (is_duplicate, existing_record)
        """
        try:
            df = pd.read_csv(self.csv_file)
            file_hash = self.calculate_file_hash(file_content)
            
            # Check for exact hash match
            hash_matches = df[df['file_hash'] == file_hash]
            if not hash_matches.empty:
                return True, hash_matches.iloc[0].to_dict()
            
            # Check for same filename (secondary check)
            name_matches = df[df['file_name'] == file_name]
            if not name_matches.empty:
                return True, name_matches.iloc[0].to_dict()
            
            return False, None
            
        except Exception as e:
            print(f"Error checking for duplicates: {e}")
            return False, None
    
    def is_duplicate_url(self, url: str, video_id: str) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a YouTube URL is a duplicate.
        
        Returns:
            Tuple of (is_duplicate, existing_record)
        """
        try:
            df = pd.read_csv(self.csv_file)
            
            # Check for video ID match
            video_matches = df[df['video_id'] == video_id]
            if not video_matches.empty:
                return True, video_matches.iloc[0].to_dict()
            
            # Check for URL hash match
            url_hash = self.calculate_url_hash(url)
            hash_matches = df[df['file_hash'] == url_hash]
            if not hash_matches.empty:
                return True, hash_matches.iloc[0].to_dict()
            
            return False, None
            
        except Exception as e:
            print(f"Error checking for URL duplicates: {e}")
            return False, None
    
    def add_document_record(self, 
                          title: str,
                          author: str,
                          summary: str,
                          doc_type: str,
                          genre: str,
                          topic: str,
                          difficulty: str,
                          source_type: str,
                          tags: str,
                          chunks: int,
                          chunk_size: int,
                          chunk_overlap: int,
                          file_content: bytes = None,
                          file_name: str = "",
                          file_size: int = 0,
                          source_url: str = "",
                          video_id: str = "") -> str:
        """
        Add a new document record to the tracker CSV.
        
        Returns:
            Document ID (UUID)
        """
        try:
            # Generate unique ID
            import uuid
            doc_id = str(uuid.uuid4())
            
            # Calculate hash based on content type
            if file_content:
                file_hash = self.calculate_file_hash(file_content)
            elif source_url:
                file_hash = self.calculate_url_hash(source_url)
            else:
                file_hash = hashlib.sha256(title.encode()).hexdigest()
            
            # Create new record
            new_record = {
                "id": doc_id,
                "title": title,
                "author": author,
                "summary": summary,
                "type": doc_type,
                "genre": genre,
                "topic": topic,
                "difficulty": difficulty,
                "source_type": source_type,
                "tags": tags,
                "chunks": chunks,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "file_hash": file_hash,
                "file_name": file_name,
                "file_size": file_size,
                "upload_date": datetime.now().isoformat(),
                "source_url": source_url,
                "video_id": video_id
            }
            
            # Read existing CSV and append new record
            df = pd.read_csv(self.csv_file)
            new_df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            new_df.to_csv(self.csv_file, index=False)
            
            print(f"✅ Added document record: {title} (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            print(f"❌ Error adding document record: {e}")
            return ""
    
    def get_all_documents(self) -> pd.DataFrame:
        """Get all documents from the tracker."""
        try:
            return pd.read_csv(self.csv_file)
        except Exception as e:
            print(f"Error reading tracker CSV: {e}")
            return pd.DataFrame()
    
    def get_document_stats(self) -> Dict:
        """Get statistics about tracked documents."""
        try:
            df = pd.read_csv(self.csv_file)
            
            if df.empty:
                return {"total": 0, "by_type": {}, "by_source": {}, "total_chunks": 0}
            
            stats = {
                "total": len(df),
                "by_type": df['type'].value_counts().to_dict(),
                "by_source": df['source_type'].value_counts().to_dict(),
                "total_chunks": df['chunks'].sum() if 'chunks' in df.columns else 0,
                "latest_upload": df['upload_date'].max() if 'upload_date' in df.columns else "Unknown"
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting document stats: {e}")
            return {"total": 0, "by_type": {}, "by_source": {}, "total_chunks": 0}
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document record from the tracker."""
        try:
            df = pd.read_csv(self.csv_file)
            df_filtered = df[df['id'] != doc_id]
            
            if len(df_filtered) < len(df):
                df_filtered.to_csv(self.csv_file, index=False)
                print(f"✅ Removed document: {doc_id}")
                return True
            else:
                print(f"⚠️ Document not found: {doc_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error removing document: {e}")
            return False
    
    def search_documents(self, query: str) -> pd.DataFrame:
        """Search documents by title, author, or tags."""
        try:
            df = pd.read_csv(self.csv_file)
            
            if df.empty:
                return df
            
            # Search in multiple columns
            mask = (
                df['title'].str.contains(query, case=False, na=False) |
                df['author'].str.contains(query, case=False, na=False) |
                df['tags'].str.contains(query, case=False, na=False) |
                df['topic'].str.contains(query, case=False, na=False)
            )
            
            return df[mask]
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return pd.DataFrame()
    
    def export_metadata_csv(self, output_file: str = "documents_metadata_export.csv") -> bool:
        """Export a simplified metadata CSV for compatibility with existing systems."""
        try:
            df = pd.read_csv(self.csv_file)
            
            # Create simplified export with standard columns
            export_df = df[['title', 'chunks', 'author', 'summary', 'type', 'genre', 'topic', 'difficulty', 'source_type', 'tags']].copy()
            export_df.to_csv(output_file, index=False)
            
            print(f"✅ Exported metadata to: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting metadata: {e}")
            return False


# Global instance for use throughout the application
document_tracker = DocumentTracker() 