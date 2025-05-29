"""
Enhanced Supabase Vector Store with dedicated metadata columns for better performance.
This maintains compatibility with LangChain while adding dedicated columns for fast filtering.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from supabase.client import Client


class EnhancedSupabaseVectorStore(SupabaseVectorStore):
    """
    Enhanced Supabase Vector Store that supports both JSON metadata and dedicated columns.
    
    This class extends the standard SupabaseVectorStore to:
    1. Maintain compatibility with LangChain's metadata handling
    2. Add dedicated columns for better query performance
    3. Support fast filtering on common metadata fields
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensure_enhanced_table()
    
    def ensure_enhanced_table(self):
        """
        Ensure the documents table has the enhanced schema with dedicated columns.
        This method checks if the enhanced columns exist and creates them if needed.
        """
        try:
            # Check if enhanced columns exist by trying to select them
            result = self.client.table(self.table_name).select(
                "id, title, author, doc_type, genre, topic, difficulty, tags, source_type, summary"
            ).limit(1).execute()
            print("✅ Enhanced columns already exist")
        except Exception as e:
            print(f"⚠️ Enhanced columns not found: {e}")
            print("📝 You need to add these columns to your Supabase table:")
            print("""
            ALTER TABLE documents 
            ADD COLUMN IF NOT EXISTS title TEXT,
            ADD COLUMN IF NOT EXISTS author TEXT,
            ADD COLUMN IF NOT EXISTS doc_type TEXT,
            ADD COLUMN IF NOT EXISTS genre TEXT,
            ADD COLUMN IF NOT EXISTS topic TEXT,
            ADD COLUMN IF NOT EXISTS difficulty TEXT,
            ADD COLUMN IF NOT EXISTS tags TEXT,
            ADD COLUMN IF NOT EXISTS source_type TEXT,
            ADD COLUMN IF NOT EXISTS summary TEXT;
            
            -- Add indexes for better performance
            CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);
            CREATE INDEX IF NOT EXISTS idx_documents_author ON documents(author);
            CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(doc_type);
            CREATE INDEX IF NOT EXISTS idx_documents_genre ON documents(genre);
            CREATE INDEX IF NOT EXISTS idx_documents_difficulty ON documents(difficulty);
            CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);
            """)
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        Add documents to the vector store with both JSON metadata and dedicated columns.
        
        Args:
            documents: List of Document objects to add
            **kwargs: Additional arguments passed to parent method
            
        Returns:
            List of document IDs
        """
        # First, add documents using the standard LangChain method
        doc_ids = super().add_documents(documents, **kwargs)
        
        # Then update the dedicated columns for better performance
        self._update_dedicated_columns(documents, doc_ids)
        
        return doc_ids
    
    def _update_dedicated_columns(self, documents: List[Document], doc_ids: List[str]):
        """
        Update the dedicated metadata columns for the inserted documents.
        
        Args:
            documents: List of Document objects
            doc_ids: List of corresponding document IDs
        """
        try:
            for doc, doc_id in zip(documents, doc_ids):
                metadata = doc.metadata
                
                # Extract metadata fields with defaults
                update_data = {
                    "title": metadata.get("title", "Unknown"),
                    "author": metadata.get("author", "Unknown"),
                    "doc_type": metadata.get("type", "Unknown"),
                    "genre": metadata.get("genre", "Unknown"),
                    "topic": metadata.get("topic", "Unknown"),
                    "difficulty": metadata.get("difficulty", "Unknown"),
                    "tags": metadata.get("tags", ""),
                    "source_type": metadata.get("source_type", "Unknown"),
                    "summary": metadata.get("summary", "")
                }
                
                # Update the dedicated columns
                result = self.client.table(self.table_name).update(update_data).eq("id", doc_id).execute()
                
            print(f"✅ Updated dedicated columns for {len(documents)} documents")
            
        except Exception as e:
            print(f"⚠️ Error updating dedicated columns: {e}")
            print("Documents were still added with JSON metadata, but dedicated columns may not be populated")
    
    def search_by_metadata(self, filters: Dict[str, Any], limit: int = 10) -> List[Dict]:
        """
        Fast search using dedicated columns instead of JSON metadata.
        
        Args:
            filters: Dictionary of metadata filters (e.g., {"doc_type": "Book", "difficulty": "Intermediate"})
            limit: Maximum number of results to return
            
        Returns:
            List of document metadata dictionaries
        """
        try:
            query = self.client.table(self.table_name).select("*")
            
            # Apply filters using dedicated columns for fast performance
            for field, value in filters.items():
                if field == "type":
                    query = query.eq("doc_type", value)
                elif field in ["title", "author", "genre", "topic", "difficulty", "source_type"]:
                    query = query.eq(field, value)
                elif field == "tags":
                    query = query.ilike("tags", f"%{value}%")
            
            result = query.limit(limit).execute()
            return result.data
            
        except Exception as e:
            print(f"Error in metadata search: {e}")
            return []
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about documents using dedicated columns for fast aggregation.
        
        Returns:
            Dictionary with document statistics
        """
        try:
            # Get total count
            total_result = self.client.table(self.table_name).select("id", count="exact").execute()
            total_docs = total_result.count
            
            # Get counts by type using dedicated columns
            type_result = self.client.table(self.table_name).select("doc_type").execute()
            type_counts = {}
            for row in type_result.data:
                doc_type = row.get("doc_type", "Unknown")
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            # Get counts by difficulty
            difficulty_result = self.client.table(self.table_name).select("difficulty").execute()
            difficulty_counts = {}
            for row in difficulty_result.data:
                difficulty = row.get("difficulty", "Unknown")
                difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            
            return {
                "total_documents": total_docs,
                "by_type": type_counts,
                "by_difficulty": difficulty_counts
            }
            
        except Exception as e:
            print(f"Error getting document stats: {e}")
            return {"total_documents": 0, "by_type": {}, "by_difficulty": {}}


def create_enhanced_vector_store(supabase_client: Client, embeddings, table_name: str = "documents_enhanced"):
    """
    Create an enhanced vector store with dedicated metadata columns.
    
    Args:
        supabase_client: Supabase client instance
        embeddings: Embeddings model instance
        table_name: Name of the table to use (default: "documents_enhanced")
        
    Returns:
        EnhancedSupabaseVectorStore instance
    """
    return EnhancedSupabaseVectorStore(
        embedding=embeddings,
        client=supabase_client,
        table_name=table_name,
        query_name=f"match_{table_name}",
    ) 