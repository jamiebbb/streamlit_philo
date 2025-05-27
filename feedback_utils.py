import os
from datetime import datetime
import streamlit as st
import uuid

try:
    import numpy as np
except ImportError:
    st.error("NumPy is required for embedding-based feedback. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "numpy"])
    import numpy as np

class FeedbackHandler:
    def __init__(self, supabase_client):
        """Initialize the feedback handler with a Supabase client."""
        self.supabase = supabase_client
        self.ensure_feedback_table()
        
    def ensure_feedback_table(self):
        """Check if feedback table exists, create it if not."""
        try:
            # Check if table exists by trying to select from it
            self.supabase.table("feedback").select("id").limit(1).execute()
            print("Feedback table exists")
        except Exception as e:
            print(f"Error checking feedback table: {e}")
            print("You may need to create the feedback table manually. Execute this SQL in Supabase:")
            print("""
            create table feedback (
                id uuid primary key default uuid_generate_v4(),
                query text,
                response text,
                feedback text,
                metadata jsonb,
                user_id text,
                timestamp timestamp with time zone default now()
            );
            """)
    
    def add_feedback_buttons(self, user_query, ai_response, chat_id=None):
        """Add feedback buttons to the Streamlit interface."""
        # Create a unique ID for this feedback if not provided
        if not chat_id:
            if "chat_id" not in st.session_state:
                st.session_state.chat_id = str(uuid.uuid4())
            chat_id = st.session_state.chat_id
            
        # Create columns for buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        # Add feedback buttons
        with col1:
            if st.button("👍 Helpful", key=f"helpful_{hash(ai_response)}"):
                self.store_feedback(user_query, ai_response, "helpful", chat_id)
                st.success("Thanks for your feedback!")
                
        with col2:
            if st.button("👎 Not Helpful", key=f"not_helpful_{hash(ai_response)}"):
                self.store_feedback(user_query, ai_response, "not_helpful", chat_id)
                st.error("We'll try to do better next time!")
                
        with col3:
            if st.button("🤔 Partially Helpful", key=f"partial_{hash(ai_response)}"):
                self.store_feedback(user_query, ai_response, "partial", chat_id)
                st.info("Thanks for your feedback!")
    
    def add_detailed_feedback(self, user_query, ai_response, chat_id=None):
        """Add a detailed feedback form."""
        with st.expander("Provide detailed feedback"):
            rating = st.slider("Rate this response (1-5)", 1, 5, 3)
            comment = st.text_area("Additional comments (optional)")
            
            if st.button("Submit Detailed Feedback"):
                self.store_detailed_feedback(user_query, ai_response, rating, comment, chat_id)
                st.success("Thank you for your detailed feedback!")
    
    def store_feedback(self, query, response, feedback_type, chat_id=None):
        """Store basic feedback in Supabase."""
        feedback_data = {
            "query": query,
            "response": response,
            "feedback": feedback_type,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "chat_id": chat_id or str(uuid.uuid4()),
                "user_id": st.session_state.get("user_id", "anonymous")
            }
        }
        
        try:
            result = self.supabase.table("feedback").insert(feedback_data).execute()
            print(f"Feedback stored: {feedback_type}")
            return True
        except Exception as e:
            print(f"Error storing feedback: {e}")
            return False
    
    def store_detailed_feedback(self, query, response, rating, comment, chat_id=None):
        """Store detailed feedback in Supabase."""
        feedback_data = {
            "query": query,
            "response": response,
            "feedback": "detailed",
            "metadata": {
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.now().isoformat(),
                "chat_id": chat_id or str(uuid.uuid4()),
                "user_id": st.session_state.get("user_id", "anonymous")
            }
        }
        
        try:
            result = self.supabase.table("feedback").insert(feedback_data).execute()
            print(f"Detailed feedback stored: rating={rating}")
            return True
        except Exception as e:
            print(f"Error storing detailed feedback: {e}")
            return False
    
    def get_feedback_stats(self):
        """Get feedback statistics for display."""
        try:
            result = self.supabase.table("feedback").select("feedback").execute()
            feedback_data = result.data
            
            stats = {
                "helpful": 0,
                "not_helpful": 0,
                "partial": 0,
                "detailed": 0
            }
            
            for item in feedback_data:
                feedback_type = item.get("feedback", "unknown")
                if feedback_type in stats:
                    stats[feedback_type] += 1
            
            return stats
        except Exception as e:
            print(f"Error getting feedback stats: {e}")
            return None
            
    def get_relevant_feedback(self, query, similarity_threshold=0.75, embeddings_model=None):
        """Retrieve relevant feedback for a given query using embeddings.
        
        This method searches for past feedback that might be relevant to the current query,
        using semantic similarity via embeddings instead of keyword matching.
        
        Args:
            query: The current user query
            similarity_threshold: Minimum similarity score to consider feedback relevant (0.0-1.0)
            embeddings_model: OpenAI embeddings model instance
            
        Returns:
            A list of relevant feedback entries with corrections
        """
        try:
            # Get all detailed feedback with comments
            result = self.supabase.table("feedback")\
                .select("query, response, metadata")\
                .eq("feedback", "detailed")\
                .execute()
                
            detailed_feedback = result.data
            relevant_feedback = []
            
            if not detailed_feedback:
                return relevant_feedback
            
            # If embeddings model is provided, use semantic similarity
            if embeddings_model:
                try:
                    # Get embedding for current query
                    query_embedding = embeddings_model.embed_query(query)
                    
                    for item in detailed_feedback:
                        try:
                            past_query = item.get("query", "")
                            metadata = item.get("metadata", {})
                            comment = metadata.get("comment", "")
                            rating = metadata.get("rating", 0)
                            
                            # Only process feedback with actual comments and low ratings
                            if comment and rating <= 3 and past_query:
                                # Get embedding for past query
                                past_query_embedding = embeddings_model.embed_query(past_query)
                                
                                # Calculate cosine similarity
                                similarity = self._cosine_similarity(query_embedding, past_query_embedding)
                                
                                if similarity >= similarity_threshold:
                                    relevant_feedback.append({
                                        "past_query": past_query,
                                        "past_response": item.get("response"),
                                        "comment": comment,
                                        "rating": rating,
                                        "similarity": similarity
                                    })
                        except Exception as e:
                            print(f"Error processing individual feedback item: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Error with embeddings, falling back to keyword matching: {e}")
                    # Fall back to keyword matching
                    embeddings_model = None
            
            # Fallback to keyword matching if no embeddings model provided or embeddings failed
            if not embeddings_model:
                query_keywords = set(query.lower().split())
                
                for item in detailed_feedback:
                    try:
                        past_query = item.get("query", "").lower()
                        past_keywords = set(past_query.split())
                        common_keywords = query_keywords.intersection(past_keywords)
                        
                        # Calculate simple similarity score
                        similarity = len(common_keywords) / max(len(query_keywords), len(past_keywords)) if query_keywords or past_keywords else 0
                        
                        if similarity >= similarity_threshold:
                            metadata = item.get("metadata", {})
                            comment = metadata.get("comment", "")
                            rating = metadata.get("rating", 0)
                            
                            if comment and rating <= 3:
                                relevant_feedback.append({
                                    "past_query": item.get("query"),
                                    "past_response": item.get("response"),
                                    "comment": comment,
                                    "rating": rating,
                                    "similarity": similarity
                                })
                    except Exception as e:
                        print(f"Error processing feedback item in keyword mode: {e}")
                        continue
            
            # Sort by similarity
            relevant_feedback.sort(key=lambda x: x["similarity"], reverse=True)
            return relevant_feedback
            
        except Exception as e:
            print(f"Error retrieving relevant feedback: {e}")
            return []
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        try:
            # Convert to numpy arrays
            vec1 = np.array(vec1, dtype=float)
            vec2 = np.array(vec2, dtype=float)
            
            # Check for valid vectors
            if len(vec1) == 0 or len(vec2) == 0 or len(vec1) != len(vec2):
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure similarity is between 0 and 1
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0 