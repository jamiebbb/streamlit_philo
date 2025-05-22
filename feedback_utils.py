import os
from datetime import datetime
import streamlit as st
import uuid

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