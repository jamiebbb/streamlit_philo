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
            # Check if enhanced columns exist by trying to select them (silent check)
            result = self._client.table(self.table_name).select(
                "id, title, author, doc_type, genre, topic, difficulty, tags, source_type, summary"
            ).limit(1).execute()
            # Silent success - no print statement
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
        print(f"📊 Debug - add_documents called with table_name: {self.table_name}")
        print(f"📊 Debug - Number of documents: {len(documents)}")
        
        try:
            # For enhanced table, we'll insert directly with all columns populated
            if self.table_name == "documents_enhanced":
                print("📊 Debug - Using enhanced method for documents_enhanced table")
                return self._add_documents_enhanced(documents, **kwargs)
            else:
                print("📊 Debug - Using standard method with column updates")
                # For standard table, use the parent method and update columns
                doc_ids = super().add_documents(documents, **kwargs)
                self._update_dedicated_columns(documents, doc_ids)
                return doc_ids
        except Exception as e:
            print(f"❌ Error in add_documents: {e}")
            print(f"📊 Debug - Exception type: {type(e).__name__}")
            
            # For enhanced table, don't fall back - we want to see the error
            if self.table_name == "documents_enhanced":
                print("❌ Enhanced table insert failed - not falling back to preserve error visibility")
                raise e
            else:
                print("⚠️ Falling back to standard method")
                # Fallback to standard method
                return super().add_documents(documents, **kwargs)
    
    def _add_documents_enhanced(self, documents: List[Document], **kwargs) -> List[str]:
        """
        Add documents directly to the enhanced table with all columns populated.
        
        Args:
            documents: List of Document objects to add
            **kwargs: Additional arguments
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        
        try:
            # Get embeddings for all documents
            texts = [doc.page_content for doc in documents]
            
            # Use the correct embedding attribute from parent class
            # Check for both possible attribute names
            if hasattr(self, '_embedding'):
                embedding_model = self._embedding
            elif hasattr(self, 'embedding'):
                embedding_model = self.embedding
            else:
                raise AttributeError("No embedding model found. Expected '_embedding' or 'embedding' attribute.")
            
            embeddings = embedding_model.embed_documents(texts)
            
            # Calculate total chunks for this document batch (assuming all chunks are from same document)
            total_chunks = len(documents)
            
            # Group documents by title to handle multiple documents with same title
            documents_by_title = {}
            for i, doc in enumerate(documents):
                title = doc.metadata.get("title", "Unknown")
                if title not in documents_by_title:
                    documents_by_title[title] = []
                documents_by_title[title].append((i, doc))
            
            for doc, embedding in zip(documents, embeddings):
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)
                
                metadata = doc.metadata
                
                # Debug: Print metadata to see what we're working with
                print(f"📊 Debug - Document metadata: {metadata}")
                
                # Calculate chunk_id (1-based indexing)
                title = metadata.get("title", "Unknown")
                chunk_id = 1
                for i, (doc_idx, _) in enumerate(documents_by_title[title]):
                    if documents[doc_idx] == doc:
                        chunk_id = i + 1
                        break
                
                # Extract source URL from metadata
                source_url = self._extract_source_url(metadata)
                
                # Prepare data for insertion with all columns
                insert_data = {
                    "id": doc_id,
                    "content": doc.page_content,
                    "metadata": metadata,  # Keep JSON metadata for compatibility
                    "embedding": embedding,
                    # Dedicated columns with proper mapping
                    "title": metadata.get("title", "Unknown"),
                    "author": metadata.get("author", "Unknown"),
                    "doc_type": metadata.get("type", "Unknown"),  # Map "type" to "doc_type"
                    "genre": metadata.get("genre", "Unknown"),
                    "topic": metadata.get("topic", "Unknown"),
                    "difficulty": metadata.get("difficulty", "Unknown"),
                    "tags": metadata.get("tags", ""),
                    "source_type": metadata.get("source_type", "Unknown"),
                    "summary": metadata.get("summary", ""),
                    # New columns
                    "chunk_id": chunk_id,
                    "total_chunks": len(documents_by_title[title]),
                    "source": source_url
                }
                
                # Debug: Print what we're inserting
                print(f"📊 Debug - Insert data keys: {list(insert_data.keys())}")
                print(f"📊 Debug - Insert data values: title='{insert_data['title']}', chunk_id={insert_data['chunk_id']}, total_chunks={insert_data['total_chunks']}, source='{insert_data['source']}'")
                
                # Insert into enhanced table
                try:
                    result = self._client.table(self.table_name).insert(insert_data).execute()
                    
                    # Debug: Print result
                    print(f"📊 Debug - Insert result success: {len(result.data) > 0}")
                    if result.data:
                        inserted_doc = result.data[0]
                        print(f"📊 Debug - Inserted doc columns: title='{inserted_doc.get('title')}', chunk_id={inserted_doc.get('chunk_id')}, total_chunks={inserted_doc.get('total_chunks')}")
                    else:
                        print(f"⚠️ Warning: No data returned for document {doc_id}")
                        
                except Exception as insert_error:
                    print(f"❌ Insert error: {insert_error}")
                    # Try to get more details about the error
                    if "column" in str(insert_error).lower():
                        print(f"💡 This might be a column mismatch. Check if all columns exist in your Supabase table.")
                        print(f"💡 Expected columns: {list(insert_data.keys())}")
                    raise insert_error
            
            print(f"✅ Successfully added {len(documents)} documents to enhanced table")
            return doc_ids
            
        except Exception as e:
            print(f"❌ Error adding documents to enhanced table: {e}")
            raise e
    
    def _extract_source_url(self, metadata: Dict[str, Any]) -> str:
        """
        Extract source URL from metadata based on source type.
        
        Args:
            metadata: Document metadata dictionary
            
        Returns:
            Source URL string or empty string if not found
        """
        # Check for explicit source URL
        if "source" in metadata and metadata["source"]:
            return str(metadata["source"])
        
        # Check for URL field
        if "url" in metadata and metadata["url"]:
            return str(metadata["url"])
        
        # Check for YouTube URL
        if "youtube_url" in metadata and metadata["youtube_url"]:
            return str(metadata["youtube_url"])
        
        # Check for web URL
        if "web_url" in metadata and metadata["web_url"]:
            return str(metadata["web_url"])
        
        # Check source_type and try to construct URL
        source_type = metadata.get("source_type", "").lower()
        
        if source_type == "youtube" and "video_id" in metadata:
            return f"https://www.youtube.com/watch?v={metadata['video_id']}"
        
        if source_type == "pdf" and "file_path" in metadata:
            return metadata["file_path"]
        
        if source_type == "web" and "page_url" in metadata:
            return metadata["page_url"]
        
        # Return empty string if no source found
        return ""
    
    def _update_dedicated_columns(self, documents: List[Document], doc_ids: List[str]):
        """
        Update the dedicated metadata columns for the inserted documents.
        
        Args:
            documents: List of Document objects
            doc_ids: List of corresponding document IDs
        """
        try:
            # Group documents by title to calculate chunk information
            documents_by_title = {}
            for i, doc in enumerate(documents):
                title = doc.metadata.get("title", "Unknown")
                if title not in documents_by_title:
                    documents_by_title[title] = []
                documents_by_title[title].append((i, doc))
            
            for i, (doc, doc_id) in enumerate(zip(documents, doc_ids)):
                metadata = doc.metadata
                
                # Calculate chunk_id (1-based indexing)
                title = metadata.get("title", "Unknown")
                chunk_id = 1
                for j, (doc_idx, _) in enumerate(documents_by_title[title]):
                    if doc_idx == i:
                        chunk_id = j + 1
                        break
                
                # Extract source URL from metadata
                source_url = self._extract_source_url(metadata)
                
                # Extract metadata fields with defaults and proper mapping
                update_data = {
                    "title": metadata.get("title", "Unknown"),
                    "author": metadata.get("author", "Unknown"),
                    "doc_type": metadata.get("type", "Unknown"),  # Map "type" to "doc_type"
                    "genre": metadata.get("genre", "Unknown"),
                    "topic": metadata.get("topic", "Unknown"),
                    "difficulty": metadata.get("difficulty", "Unknown"),
                    "tags": metadata.get("tags", ""),
                    "source_type": metadata.get("source_type", "Unknown"),
                    "summary": metadata.get("summary", ""),
                    # New columns
                    "chunk_id": chunk_id,
                    "total_chunks": len(documents_by_title[title]),
                    "source": source_url
                }
                
                # Update the dedicated columns
                result = self._client.table(self.table_name).update(update_data).eq("id", doc_id).execute()
                
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
            query = self._client.table(self.table_name).select("*")
            
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
            total_result = self._client.table(self.table_name).select("id", count="exact").execute()
            total_docs = total_result.count
            
            # Get counts by type using dedicated columns
            type_result = self._client.table(self.table_name).select("doc_type").execute()
            type_counts = {}
            for row in type_result.data:
                doc_type = row.get("doc_type", "Unknown")
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            # Get counts by difficulty
            difficulty_result = self._client.table(self.table_name).select("difficulty").execute()
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

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs
    ) -> List[tuple]:
        """
        Override parent method to handle correct function parameter order.
        
        Args:
            embedding: Query embedding vector
            k: Number of documents to return
            **kwargs: Additional arguments
            
        Returns:
            List of (Document, score) tuples
        """
        print(f"🔍 VECTOR DEBUG: similarity_search_with_score_by_vector called with k={k}")
        print(f"🔍 VECTOR DEBUG: Function name: {self.query_name}")
        print(f"🔍 VECTOR DEBUG: Table name: {self.table_name}")
        
        try:
            # Call the Supabase function with correct parameter order
            # Your function expects: match_documents_enhanced(match_count, match_threshold, query_embedding)
            match_threshold = kwargs.get("match_threshold", 0.0)  # Changed from 0.78 to 0.0 to get top K regardless of similarity
            
            print(f"🔍 VECTOR DEBUG: Trying first parameter order - match_count={k}, match_threshold={match_threshold}")
            
            result = self._client.rpc(
                self.query_name,
                {
                    "match_count": k,
                    "match_threshold": match_threshold,
                    "query_embedding": embedding
                }
            ).execute()
            
            print(f"🔍 VECTOR DEBUG: First attempt successful! Got {len(result.data)} results")
            
            docs_with_scores = []
            for i, row in enumerate(result.data):
                print(f"🔍 VECTOR DEBUG: Processing row {i+1}: title='{row.get('title', 'No title')}', similarity={row.get('similarity', 0.0)}")
                
                # Create Document object
                doc = Document(
                    page_content=row["content"],
                    metadata=row.get("metadata", {})
                )
                
                # Add enhanced metadata to the document metadata
                doc.metadata.update({
                    "title": row.get("title"),
                    "author": row.get("author"),
                    "doc_type": row.get("doc_type"),
                    "genre": row.get("genre"),
                    "topic": row.get("topic"),
                    "difficulty": row.get("difficulty"),
                    "tags": row.get("tags"),
                    "source_type": row.get("source_type"),
                    "summary": row.get("summary"),
                    "chunk_id": row.get("chunk_id"),
                    "total_chunks": row.get("total_chunks"),
                    "source": row.get("source")
                })
                
                # Get similarity score
                score = row.get("similarity", 0.0)
                docs_with_scores.append((doc, score))
            
            print(f"🔍 VECTOR DEBUG: Returning {len(docs_with_scores)} documents with scores")
            return docs_with_scores
            
        except Exception as e:
            print(f"🔍 VECTOR DEBUG: First attempt failed: {e}")
            print(f"🔍 VECTOR DEBUG: Function name: {self.query_name}")
            print(f"🔍 VECTOR DEBUG: Parameters: match_count={k}, match_threshold={match_threshold}")
            
            # Try alternative parameter order if the first one fails
            try:
                print("🔍 VECTOR DEBUG: Trying alternative parameter order...")
                result = self._client.rpc(
                    self.query_name,
                    {
                        "query_embedding": embedding,
                        "match_count": k,
                        "match_threshold": match_threshold  # Use the same threshold as above
                    }
                ).execute()
                
                print(f"🔍 VECTOR DEBUG: Alternative attempt successful! Got {len(result.data)} results")
                
                docs_with_scores = []
                for row in result.data:
                    doc = Document(
                        page_content=row["content"],
                        metadata=row.get("metadata", {})
                    )
                    
                    # Add enhanced metadata
                    doc.metadata.update({
                        "title": row.get("title"),
                        "author": row.get("author"),
                        "doc_type": row.get("doc_type"),
                        "genre": row.get("genre"),
                        "topic": row.get("topic"),
                        "difficulty": row.get("difficulty"),
                        "tags": row.get("tags"),
                        "source_type": row.get("source_type"),
                        "summary": row.get("summary"),
                        "chunk_id": row.get("chunk_id"),
                        "total_chunks": row.get("total_chunks"),
                        "source": row.get("source")
                    })
                    
                    score = row.get("similarity", 0.0)
                    docs_with_scores.append((doc, score))
                
                print("✅ Alternative parameter order worked!")
                return docs_with_scores
                
            except Exception as e2:
                print(f"🔍 VECTOR DEBUG: Both parameter orders failed: {e2}")
                return []

    def similarity_search(
        self, query: str, k: int = 4, **kwargs
    ) -> List[Document]:
        """
        Override the main similarity_search method to use our custom function.
        
        Args:
            query: Query string
            k: Number of documents to return
            **kwargs: Additional arguments
            
        Returns:
            List of Document objects
        """
        print(f"🔍 VECTOR DEBUG: similarity_search called with query='{query}', k={k}")
        
        try:
            # Get embedding for the query
            if hasattr(self, '_embedding'):
                embedding_model = self._embedding
                print(f"🔍 VECTOR DEBUG: Using _embedding attribute")
            elif hasattr(self, 'embedding'):
                embedding_model = self.embedding
                print(f"🔍 VECTOR DEBUG: Using embedding attribute")
            else:
                raise AttributeError("No embedding model found")
            
            print(f"🔍 VECTOR DEBUG: Generating embedding for query...")
            query_embedding = embedding_model.embed_query(query)
            print(f"🔍 VECTOR DEBUG: Generated embedding with {len(query_embedding)} dimensions")
            
            # Use our custom similarity search with score
            print(f"🔍 VECTOR DEBUG: Calling similarity_search_with_score_by_vector...")
            docs_with_scores = self.similarity_search_with_score_by_vector(
                query_embedding, k=k, **kwargs
            )
            
            print(f"🔍 VECTOR DEBUG: Got {len(docs_with_scores)} docs with scores")
            
            # Return just the documents (without scores)
            docs = [doc for doc, score in docs_with_scores]
            print(f"🔍 VECTOR DEBUG: Returning {len(docs)} documents")
            return docs
            
        except Exception as e:
            print(f"🔍 VECTOR DEBUG: Error in similarity_search: {e}")
            # Fallback to parent method if our custom method fails
            try:
                print("🔍 VECTOR DEBUG: Falling back to parent similarity_search method...")
                return super().similarity_search(query, k=k, **kwargs)
            except Exception as e2:
                print(f"🔍 VECTOR DEBUG: Parent method also failed: {e2}")
                return []


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