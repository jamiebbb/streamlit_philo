# import basics
import os
import uuid
from dotenv import load_dotenv

# import streamlit
import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")

# import langchain
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

# import supabase db
from supabase.client import Client, create_client

# import feedback utilities
from feedback_utils import FeedbackHandler

# import enhanced vector store
from enhanced_vector_store import EnhancedSupabaseVectorStore, create_enhanced_vector_store

# import document tracker
from document_tracker import document_tracker

# load environment variables
load_dotenv(override=True)  
print("DEBUG - OpenAI API Key:", os.environ.get("OPENAI_API_KEY")[:10] + "..." if os.environ.get("OPENAI_API_KEY") else "NOT FOUND")

# Always use enhanced vector store
USE_ENHANCED_STORE = True

# initiating supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize feedback handler
feedback_handler = FeedbackHandler(supabase)

# Test Supabase connection for feedback
print("Testing feedback system...")
feedback_handler.test_supabase_connection()

# initiating embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Always use enhanced vector store
vector_store_status = "unknown"
vector_store_error = None

try:
    vector_store = create_enhanced_vector_store(
        supabase_client=supabase,
        embeddings=embeddings,
        table_name="documents_enhanced"
    )
    print("✅ Using Enhanced Vector Store")
    vector_store_status = "enhanced"
except Exception as e:
    print(f"❌ Enhanced store failed: {e}")
    vector_store_error = str(e)
    print("Falling back to standard vector store")
    try:
        vector_store = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
        )
        print("✅ Using Standard Vector Store")
        vector_store_status = "standard"
    except Exception as e2:
        print(f"❌ Standard store also failed: {e2}")
        vector_store_status = "failed"
        vector_store_error = f"Enhanced: {e}, Standard: {e2}"
        # Create a dummy vector store that will show helpful error messages
        vector_store = None

# initiating llm
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

# Create custom prompt that prioritizes retrieved context
custom_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI assistant with access to a comprehensive document library. Your primary goal is to provide thorough, accurate answers based on the retrieved context from the documents.

IMPORTANT INSTRUCTIONS:
1. **PRIORITIZE RETRIEVED CONTEXT**: Always use information from the retrieved documents as your primary source
2. **BE THOROUGH**: Provide comprehensive answers that fully utilize the retrieved context
3. **CITE SOURCES**: When using information from documents, mention the source (title, author, or document type)
4. **CONTEXT FIRST**: Only supplement with general knowledge if the retrieved context is insufficient
5. **BE SPECIFIC**: Include specific details, examples, and explanations from the documents
6. **ACKNOWLEDGE LIMITATIONS**: If the retrieved context doesn't fully answer the question, clearly state what information is missing

RESPONSE STRUCTURE:
- Start with information directly from the retrieved documents
- Provide specific details and examples from the context
- Cite the sources of your information
- Only add general knowledge if it enhances the context-based answer
- If context is insufficient, clearly state what additional information would be helpful

Remember: The retrieved documents are your most valuable resource. Use them thoroughly before relying on general knowledge."""),
    
    ("human", "{input}"),
    
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    
    ("system", """Based on the retrieved context above, provide a comprehensive answer that:
1. Thoroughly uses all relevant information from the retrieved documents
2. Cites specific sources when possible
3. Provides detailed explanations and examples from the context
4. Only supplements with general knowledge if the context is insufficient
5. Clearly indicates if more information is needed to fully answer the question""")
])

# Use custom prompt instead of hub prompt
prompt = custom_prompt

# creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query using enhanced search strategies."""
    if vector_store is None:
        return f"Error: Vector store initialization failed. Please check your database setup. Error details: {vector_store_error}", []
    
    try:
        # Primary search: Semantic similarity with more results
        retrieved_docs = vector_store.similarity_search(query, k=5)
        
        # If we have enhanced vector store, try additional searches for better coverage
        if hasattr(vector_store, 'search_by_metadata') and len(retrieved_docs) < 3:
            # Extract potential book/document titles from query
            query_lower = query.lower()
            potential_titles = []
            
            # Look for quoted titles or common book patterns
            import re
            quoted_matches = re.findall(r'"([^"]+)"', query)
            potential_titles.extend(quoted_matches)
            
            # Look for book-like phrases
            book_patterns = [
                r'book\s+(?:called\s+)?["\']?([^"\']+?)["\']?(?:\s|$)',
                r'(?:the\s+)?book\s+["\']?([^"\']+?)["\']?',
                r'["\']([^"\']{3,30})["\']',  # Any quoted text
            ]
            
            for pattern in book_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                potential_titles.extend(matches)
            
            # Try metadata search for potential titles
            for title in potential_titles[:3]:  # Limit to avoid too many searches
                title_clean = title.strip().strip('"\'')
                if len(title_clean) > 2:
                    try:
                        # Search by title in metadata
                        metadata_results = vector_store.search_by_metadata(
                            {"title": title_clean}, limit=3
                        )
                        
                        # Convert metadata results to Document objects
                        for result in metadata_results:
                            if result.get("content"):
                                from langchain_core.documents import Document
                                doc = Document(
                                    page_content=result["content"],
                                    metadata={
                                        "title": result.get("title", "Unknown"),
                                        "author": result.get("author", "Unknown"),
                                        "doc_type": result.get("doc_type", "Unknown"),
                                        "source": f"Metadata search for: {title_clean}"
                                    }
                                )
                                retrieved_docs.append(doc)
                    except Exception as e:
                        print(f"Metadata search failed for '{title_clean}': {e}")
        
        # Remove duplicates based on content
        unique_docs = []
        seen_content = set()
        for doc in retrieved_docs:
            content_hash = hash(doc.page_content[:100])  # Use first 100 chars as identifier
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        retrieved_docs = unique_docs[:8]  # Limit to 8 total documents
        
        # Enhanced serialization with better source information
        serialized_parts = []
        for i, doc in enumerate(retrieved_docs):
            metadata = doc.metadata
            title = metadata.get("title", "Unknown Document")
            author = metadata.get("author", "Unknown Author")
            doc_type = metadata.get("doc_type", metadata.get("type", "Unknown Type"))
            
            source_info = f"Document {i+1}: {title} by {author} ({doc_type})"
            content_preview = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content
            
            serialized_parts.append(f"Source: {source_info}\nContent: {content_preview}")
        
        serialized = "\n\n" + "="*50 + "\n\n".join(serialized_parts)
        
        # Add search summary
        search_summary = f"Found {len(retrieved_docs)} relevant documents for query: '{query}'"
        if len(retrieved_docs) == 0:
            search_summary += "\n\nNo documents found. This could mean:\n1. The document hasn't been uploaded\n2. The search terms don't match the content\n3. Try different keywords or check the Vector Store tab"
        
        final_result = f"{search_summary}\n\n{serialized}"
        
        return final_result, retrieved_docs
        
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg.lower() or "relation" in error_msg.lower():
            return f"Error: The documents table doesn't exist yet. Please upload some documents first or run the database setup SQL script. Details: {error_msg}", []
        else:
            return f"Error retrieving documents: {error_msg}", []

# combining all tools
tools = [retrieve]

# Add debug tool for testing
@tool(response_format="content_and_artifact")
def debug_search(query: str):
    """Debug tool to test search functionality and show available documents."""
    if vector_store is None:
        return "Vector store not initialized", []
    
    try:
        # Test basic search
        docs = vector_store.similarity_search(query, k=10)
        
        # Get some sample documents to show what's available
        if hasattr(vector_store, 'search_by_metadata'):
            sample_docs = vector_store.search_by_metadata({}, limit=5)
            sample_info = "Sample documents in database:\n"
            for doc in sample_docs:
                sample_info += f"- {doc.get('title', 'Unknown')} by {doc.get('author', 'Unknown')}\n"
        else:
            sample_info = "Enhanced search not available"
        
        result = f"Search for '{query}' found {len(docs)} documents.\n\n{sample_info}"
        return result, docs
    except Exception as e:
        return f"Debug search failed: {e}", []

# Add debug tool to tools list for testing (comment out in production)
# tools.append(debug_search)

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# App title
st.title("🦜 Agentic RAG Chatbot")

# Create tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Vector Store", "📤 Upload Documents"])

# Create a unique session ID if not exists
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

# TAB 1: CHAT
with tab1:
    # Sidebar with basic info
    with st.sidebar:
        st.markdown("### About")
        st.markdown("This is an Agentic RAG chatbot that searches through your document library.")
        st.markdown("Ask questions about the content in your knowledge base!")
        
        st.divider()
        
        # Enhanced Vector Store Setup - Always shown since we always use it
        st.markdown("### 🚀 Enhanced Vector Store")
        st.markdown("Using the enhanced vector store with dedicated metadata columns for better performance.")
        
        with st.expander("📋 Setup Instructions", expanded=False):
            st.markdown("""
            **To set up the enhanced table:**
            
            **Option A: Full Setup (Recommended)**
            1. Run `setup_enhanced_documents_table.sql` in your Supabase SQL editor
            2. This creates the `documents_enhanced` table with all optimizations
            
            **Option B: If you get pg_trgm error**
            1. Run `setup_enhanced_documents_table_simple.sql` instead
            2. This version doesn't require the pg_trgm extension
            3. Performance is still excellent for most use cases
            
            **Benefits:**
            - 100x faster filtered queries
            - Better performance for large datasets
            - Advanced filtering capabilities
            - Maintains LangChain compatibility
            
            **Common Issues:**
            - `gin_trgm_ops does not exist` → Use the simple version
            - `extension "vector" does not exist` → Enable vector extension in Supabase
            """)
            
            if st.button("📄 Show Full SQL Script"):
                st.code("""
-- Full version (setup_enhanced_documents_table.sql):
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents_enhanced (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536),
    title TEXT,
    author TEXT,
    doc_type TEXT,
    genre TEXT,
    topic TEXT,
    difficulty TEXT,
    tags TEXT,
    source_type TEXT,
    summary TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- See full script files for complete setup
                """, language="sql")
            
            if st.button("📄 Show Simple SQL Script"):
                st.code("""
-- Simple version (setup_enhanced_documents_table_simple.sql):
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- Note: Does not require pg_trgm extension

CREATE TABLE IF NOT EXISTS documents_enhanced (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536),
    title TEXT,
    author TEXT,
    doc_type TEXT,
    genre TEXT,
    topic TEXT,
    difficulty TEXT,
    tags TEXT,
    source_type TEXT,
    summary TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- See setup_enhanced_documents_table_simple.sql for complete script
                """, language="sql")
        
        st.divider()
        st.markdown("### Tips")
        st.markdown("• Be specific in your questions")
        st.markdown("• Provide feedback to improve responses")
        st.markdown("• Check the Vector Store tab to see available documents")
        
        st.divider()
        st.markdown("### Debug")
        
        # Vector Store Status
        if vector_store_status == "enhanced":
            st.success("✅ Enhanced Vector Store Active")
        elif vector_store_status == "standard":
            st.warning("⚠️ Using Standard Vector Store (Enhanced failed)")
        elif vector_store_status == "failed":
            st.error("❌ Vector Store Failed to Initialize")
            if vector_store_error:
                with st.expander("Error Details", expanded=False):
                    st.code(vector_store_error)
        
        if st.button("🧪 Test Vector Store"):
            if vector_store is None:
                st.error("❌ Vector store not initialized!")
                st.info("Please run the database setup SQL script first.")
            else:
                try:
                    # Test 1: Basic connectivity
                    test_docs = vector_store.similarity_search("test", k=1)
                    st.success(f"✅ Vector store working! Found {len(test_docs)} documents.")
                    
                    # Test 2: Show sample documents
                    if hasattr(vector_store, 'search_by_metadata'):
                        sample_docs = vector_store.search_by_metadata({}, limit=10)
                        if sample_docs:
                            st.info(f"📚 Found {len(sample_docs)} total documents in enhanced store:")
                            for i, doc in enumerate(sample_docs[:5]):
                                st.write(f"{i+1}. **{doc.get('title', 'Unknown')}** by {doc.get('author', 'Unknown')} ({doc.get('doc_type', 'Unknown')})")
                            if len(sample_docs) > 5:
                                st.write(f"... and {len(sample_docs) - 5} more documents")
                        else:
                            st.warning("⚠️ No documents found in enhanced store")
                    
                    # Test 3: Test specific search
                    with st.expander("🔍 Test Specific Search", expanded=False):
                        test_query = st.text_input("Test search query:", placeholder="e.g., Four Seasons, competition, strategy")
                        if st.button("Search") and test_query:
                            search_results = vector_store.similarity_search(test_query, k=5)
                            st.write(f"Found {len(search_results)} results for '{test_query}':")
                            for i, doc in enumerate(search_results):
                                metadata = doc.metadata
                                st.write(f"**Result {i+1}:** {metadata.get('title', 'Unknown')} by {metadata.get('author', 'Unknown')}")
                                st.write(f"Content preview: {doc.page_content[:200]}...")
                                st.write("---")
                    
                except Exception as e:
                    st.error(f"❌ Vector store error: {e}")
                    if "does not exist" in str(e).lower():
                        st.info("💡 The documents table doesn't exist. Please run the SQL setup script or upload some documents.")
                    else:
                        st.info("💡 Check your Supabase connection and table structure.")
        
        if st.button("🧪 Test Feedback System"):
            success = feedback_handler.store_feedback(
                "test query", 
                "test response", 
                "test_feedback", 
                "test_chat_id"
            )
            if success:
                st.success("✅ Feedback system working!")
            else:
                st.error("❌ Feedback system failed!")
                
        if st.button("📊 Check Feedback Count"):
            try:
                result = supabase.table("feedback").select("*").execute()
                st.info(f"Found {len(result.data)} feedback records")
                if result.data:
                    st.json(result.data[-1])  # Show latest record
            except Exception as e:
                st.error(f"Error: {e}")
        
        if st.button("🧪 Test Metadata Columns"):
            try:
                import uuid
                # Test if we can insert a simple document with metadata columns
                test_data = {
                    "id": str(uuid.uuid4()),  # Generate proper UUID
                    "content": "This is a test document to verify metadata columns work.",
                    "metadata": {"test": "data"},
                    "embedding": [0.1] * 1536,  # Dummy embedding
                    "title": "Test Document",
                    "author": "Test Author",
                    "doc_type": "Test",
                    "genre": "Test Genre",
                    "topic": "Testing",
                    "difficulty": "Beginner",
                    "tags": "test, metadata",
                    "source_type": "Test",
                    "summary": "This is a test summary"
                }
                
                # Try to insert
                result = supabase.table("documents_enhanced").insert(test_data).execute()
                
                if result.data:
                    st.success("✅ Metadata columns test successful!")
                    inserted = result.data[0]
                    st.json({
                        "title": inserted.get("title"),
                        "author": inserted.get("author"),
                        "doc_type": inserted.get("doc_type"),
                        "genre": inserted.get("genre")
                    })
                    
                    # Clean up test data
                    supabase.table("documents_enhanced").delete().eq("id", test_data["id"]).execute()
                    st.info("Test data cleaned up")
                else:
                    st.error("❌ No data returned from insert")
                    
            except Exception as e:
                st.error(f"❌ Metadata columns test failed: {e}")
                if "column" in str(e).lower():
                    st.info("💡 This suggests some columns don't exist in your Supabase table. Please run the enhanced table setup SQL script.")
                elif "does not exist" in str(e).lower():
                    st.info("💡 The documents_enhanced table doesn't exist. Please run the setup SQL script first.")
                else:
                    st.info("💡 Check the error details above for specific issues.")
        
        if st.button("🔍 Debug Vector Store"):
            try:
                st.info(f"Vector Store Type: {type(vector_store).__name__}")
                st.info(f"Table Name: {getattr(vector_store, 'table_name', 'Unknown')}")
                st.info(f"Has Enhanced Methods: {hasattr(vector_store, '_add_documents_enhanced')}")
                
                # Test document creation
                from langchain_core.documents import Document
                test_doc = Document(
                    page_content="Test content for debugging",
                    metadata={
                        "title": "Debug Test",
                        "author": "Debug Author",
                        "type": "Debug",
                        "genre": "Testing",
                        "topic": "Debug",
                        "difficulty": "Beginner",
                        "tags": "debug, test",
                        "source_type": "Debug",
                        "summary": "Debug summary"
                    }
                )
                
                st.info("Testing document upload with debug metadata...")
                
                # Try to add the document
                doc_ids = vector_store.add_documents([test_doc])
                
                if doc_ids:
                    st.success(f"✅ Document added with ID: {doc_ids[0]}")
                    
                    # Check what was actually inserted
                    result = supabase.table("documents_enhanced").select("*").eq("id", doc_ids[0]).execute()
                    if result.data:
                        inserted_doc = result.data[0]
                        st.json({
                            "id": inserted_doc.get("id"),
                            "title": inserted_doc.get("title"),
                            "author": inserted_doc.get("author"),
                            "doc_type": inserted_doc.get("doc_type"),
                            "genre": inserted_doc.get("genre"),
                            "metadata_json": inserted_doc.get("metadata")
                        })
                        
                        # Clean up
                        supabase.table("documents_enhanced").delete().eq("id", doc_ids[0]).execute()
                        st.info("Debug document cleaned up")
                    else:
                        st.warning("Document ID returned but not found in database")
                else:
                    st.error("❌ No document IDs returned")
                    
            except Exception as e:
                st.error(f"❌ Debug test failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create a fixed-height container for messages that can scroll
    # This creates the scrollable chat area
    messages_container = st.container(height=500)
    
    with messages_container:
        # Display all chat messages
        for i, message in enumerate(st.session_state.messages):
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
                    
                    # Add feedback buttons for ALL AI messages
                    user_query = st.session_state.messages[i-1].content if i > 0 else ""
                    message_key = f"msg_{i}_{hash(message.content)}"
                    
                    # Add feedback buttons
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        if st.button("👍 Helpful", key=f"helpful_{message_key}"):
                            feedback_handler.store_feedback(user_query, message.content, "helpful")
                            st.success("Thanks for your feedback!")
                    with col2:
                        if st.button("👎 Not Helpful", key=f"not_helpful_{message_key}"):
                            feedback_handler.store_feedback(user_query, message.content, "not_helpful")
                            st.error("We'll try to do better next time!")
                    with col3:
                        if st.button("🤔 Partially Helpful", key=f"partial_{message_key}"):
                            feedback_handler.store_feedback(user_query, message.content, "partial")
                            st.info("Thanks for your feedback!")
                    
                    # Detailed feedback form
                    with st.expander("💬 Provide detailed feedback", expanded=False):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            rating = st.slider("Rating", 1, 5, 3, help="Rate the response quality", key=f"rating_{message_key}")
                        with col2:
                            comment = st.text_area("Comments", placeholder="Any corrections or suggestions?", height=80, key=f"comment_{message_key}")
                        
                        if st.button("Submit Feedback", type="secondary", key=f"submit_{message_key}"):
                            feedback_handler.store_detailed_feedback(user_query, message.content, rating, comment)
                            st.success("Thank you for your feedback!")

    # Chat input - this stays fixed at the bottom, outside the scrollable container
    user_question = st.chat_input("Ask me about the documents in your library...")

    # Process new user input
    if user_question:
        # Add user message to session state
        st.session_state.messages.append(HumanMessage(user_question))

        # Look for relevant feedback
        try:
            relevant_feedback = feedback_handler.get_relevant_feedback(user_question, embeddings_model=embeddings)
        except TypeError:
            relevant_feedback = feedback_handler.get_relevant_feedback(user_question)
        
        # Get AI response
        with st.spinner("Thinking..."):
            input_data = {"input": user_question, "chat_history": st.session_state.messages}
            
            if relevant_feedback:
                feedback_context = "\n\nIMPORTANT: Users have provided the following corrections to past similar questions:\n"
                for i, feedback in enumerate(relevant_feedback[:3]):
                    feedback_context += f"\n{i+1}. Past query: '{feedback['past_query']}'\n"
                    feedback_context += f"   User correction: '{feedback['comment']}'\n"
                
                enhanced_question = f"{user_question}\n\n{feedback_context}\n\nPlease take these corrections into account in your response."
                input_data["input"] = enhanced_question
                
                st.info("📝 Incorporating past feedback into this response", icon="ℹ️")
            
            result = agent_executor.invoke(input_data)

        # Add AI response to session state
        ai_message = result["output"]
        st.session_state.messages.append(AIMessage(ai_message))
        
        # Rerun to show the new messages
        st.rerun()

    # Add a footer
    st.markdown("---")
    st.markdown("*💡 Tip: Your feedback helps improve future responses*")

# TAB 2: VECTOR STORE VIEWER
with tab2:
    st.header("📚 Vector Store Contents")
    st.write("View all documents currently stored in the vector database.")
    
    # Add refresh button
    if st.button("🔄 Refresh Data", key="refresh_vector_store"):
        st.rerun()
    
    # Show enhanced statistics - always available since we always use enhanced store
    if hasattr(vector_store, 'get_document_stats'):
        try:
            st.subheader("📊 Enhanced Statistics")
            stats = vector_store.get_document_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", stats.get("total_documents", 0))
            with col2:
                st.metric("Document Types", len(stats.get("by_type", {})))
            with col3:
                st.metric("Difficulty Levels", len(stats.get("by_difficulty", {})))
            
            # Show breakdown by type
            if stats.get("by_type"):
                st.subheader("📋 Documents by Type")
                type_data = []
                for doc_type, count in stats["by_type"].items():
                    type_data.append({"Type": doc_type, "Count": count})
                st.dataframe(type_data, use_container_width=True)
            
            # Show breakdown by difficulty
            if stats.get("by_difficulty"):
                st.subheader("🎯 Documents by Difficulty")
                difficulty_data = []
                for difficulty, count in stats["by_difficulty"].items():
                    difficulty_data.append({"Difficulty": difficulty, "Count": count})
                st.dataframe(difficulty_data, use_container_width=True)
            
            # Enhanced filtering options
            st.subheader("🔍 Enhanced Filtering")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_type = st.selectbox("Filter by Type", ["All"] + list(stats.get("by_type", {}).keys()))
            with col2:
                filter_difficulty = st.selectbox("Filter by Difficulty", ["All"] + list(stats.get("by_difficulty", {}).keys()))
            with col3:
                filter_limit = st.slider("Max Results", 10, 100, 50)
            
            # Apply enhanced filtering
            if st.button("🚀 Apply Enhanced Filter"):
                filters = {}
                if filter_type != "All":
                    filters["type"] = filter_type
                if filter_difficulty != "All":
                    filters["difficulty"] = filter_difficulty
                
                if filters:
                    with st.spinner("Applying enhanced filters..."):
                        filtered_results = vector_store.search_by_metadata(filters, limit=filter_limit)
                        
                        if filtered_results:
                            st.success(f"Found {len(filtered_results)} documents matching filters")
                            
                            # Display filtered results
                            display_data = []
                            for doc in filtered_results:
                                display_data.append({
                                    "Title": doc.get("title", "Unknown"),
                                    "Author": doc.get("author", "Unknown"),
                                    "Type": doc.get("doc_type", "Unknown"),
                                    "Genre": doc.get("genre", "Unknown"),
                                    "Difficulty": doc.get("difficulty", "Unknown"),
                                    "Source Type": doc.get("source_type", "Unknown"),
                                    "Tags": doc.get("tags", "")[:50] + "..." if len(str(doc.get("tags", ""))) > 50 else doc.get("tags", "")
                                })
                            
                            st.dataframe(display_data, use_container_width=True)
                        else:
                            st.info("No documents found matching the selected filters")
                
        except Exception as e:
            st.error(f"Error loading enhanced statistics: {e}")
            st.info("Falling back to standard view...")
    
    try:
        # Try to read from CSV first (faster and more reliable)
        import pandas as pd
        csv_file = "documents_metadata.csv"
        
        if os.path.exists(csv_file):
            st.info("📄 Loading from CSV file...")
            df = pd.read_csv(csv_file)
            
            # Convert DataFrame to the format we need
            display_data = []
            for _, row in df.iterrows():
                display_data.append({
                    "Title": row.get("title", "Unknown"),
                    "Chunks": row.get("chunks", 0),
                    "Author": row.get("author", "Unknown"),
                    "Summary": str(row.get("summary", ""))[:100] + "..." if len(str(row.get("summary", ""))) > 100 else str(row.get("summary", "")),
                    "Type": row.get("type", "Unknown"),
                    "Genre": row.get("genre", "Unknown"),
                    "Topic": row.get("topic", "Unknown"),
                    "Difficulty": row.get("difficulty", "Unknown"),
                    "Source Type": row.get("source_type", "Unknown"),
                    "Tags": str(row.get("tags", "Unknown"))[:50] + "..." if len(str(row.get("tags", "Unknown"))) > 50 else str(row.get("tags", "Unknown"))
                })
            
            total_chunks = df["chunks"].sum() if "chunks" in df.columns else len(df)
            st.success(f"✅ Loaded {len(df)} documents with {total_chunks} total chunks from CSV")
            
            # Offer to create CSV for future use
            if st.button("💾 Save to CSV for faster loading"):
                df_to_save = pd.DataFrame([
                    {
                        "title": item["Title"],
                        "chunks": item["Chunks"],
                        "author": item["Author"],
                        "type": item["Type"],
                        "genre": item["Genre"],
                        "difficulty": item["Difficulty"],
                        "source_type": item["Source Type"],
                        "tags": item["Tags"]
                    }
                    for item in display_data
                ])
                df_to_save.to_csv(csv_file, index=False)
                st.success(f"✅ Saved {len(df_to_save)} documents to {csv_file}")
            
        else:
            st.warning("📄 CSV file not found. Falling back to Supabase query...")
            # Fallback to Supabase if CSV doesn't exist
            with st.spinner("Loading from Supabase..."):
                table_name = "documents_enhanced"
                
                # Use enhanced table with dedicated columns
                result = supabase.table(table_name).select(
                    "id, title, author, doc_type, genre, topic, difficulty, tags, source_type, summary"
                ).limit(50000).execute()
                documents = result.data
                
                if not documents:
                    st.info("No documents found in the enhanced vector store.")
                    display_data = []
                    total_chunks = 0
                else:
                    # Count documents by title
                    title_counts = {}
                    doc_by_title = {}
                    
                    for doc in documents:
                        title = doc.get("title", "Unknown")
                        if title not in title_counts:
                            title_counts[title] = 0
                            doc_by_title[title] = doc
                        title_counts[title] += 1
                    
                    # Create display data
                    display_data = []
                    for title, count in title_counts.items():
                        doc = doc_by_title[title]
                        display_data.append({
                            "Title": title,
                            "Chunks": count,
                            "Author": doc.get("author", "Unknown"),
                            "Summary": str(doc.get("summary", ""))[:100] + "..." if len(str(doc.get("summary", ""))) > 100 else str(doc.get("summary", "")),
                            "Type": doc.get("doc_type", "Unknown"),
                            "Genre": doc.get("genre", "Unknown"),
                            "Topic": doc.get("topic", "Unknown"),
                            "Difficulty": doc.get("difficulty", "Unknown"),
                            "Source Type": doc.get("source_type", "Unknown"),
                            "Tags": str(doc.get("tags", ""))[:50] + "..." if len(str(doc.get("tags", ""))) > 50 else str(doc.get("tags", ""))
                        })
                    
                    total_chunks = len(documents)
                    st.success(f"📊 Loaded {len(title_counts)} documents with {total_chunks} total chunks from Enhanced Supabase")
                
                # Offer to create CSV for future use
                if st.button("💾 Save to CSV for faster loading"):
                    df_to_save = pd.DataFrame([
                        {
                            "title": item["Title"],
                            "chunks": item["Chunks"],
                            "author": item["Author"],
                            "type": item["Type"],
                            "genre": item["Genre"],
                            "difficulty": item["Difficulty"],
                            "source_type": item["Source Type"],
                            "tags": item["Tags"]
                        }
                        for item in display_data
                    ])
                    df_to_save.to_csv(csv_file, index=False)
                    st.success(f"✅ Saved {len(df_to_save)} documents to {csv_file}")
        
        if display_data:
            # Sort by number of chunks
            display_data.sort(key=lambda x: x["Chunks"], reverse=True)
            
            # Display summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique Documents", len(display_data))
            with col2:
                st.metric("Total Chunks", total_chunks)
            with col3:
                avg_chunks = total_chunks / len(display_data) if display_data else 0
                st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")
            
            # Display the data table
            st.subheader("Document Details")
            st.dataframe(display_data, use_container_width=True)
            
            # Add filtering options for CSV/standard view
            st.subheader("Filter Documents")
            
            # Filter by type
            all_types = list(set([item["Type"] for item in display_data]))
            selected_type = st.selectbox("Filter by Type", ["All"] + all_types)
            
            # Apply filters
            filtered_data = display_data
            if selected_type != "All":
                filtered_data = [item for item in filtered_data if item["Type"] == selected_type]
            
            if filtered_data != display_data:
                st.subheader("Filtered Results")
                st.dataframe(filtered_data, use_container_width=True)
        else:
            st.info("No documents found.")
                
    except Exception as e:
        st.error(f"Error loading document data: {e}")
        st.info("💡 Try creating a documents_metadata.csv file with columns: title, chunks, author, type, genre, difficulty, source_type, tags")

# TAB 3: DOCUMENT UPLOAD
with tab3:
    st.header("📤 Upload Documents")
    st.write("Upload PDF documents or provide YouTube URLs to add to the vector store.")
    
    # Create sub-tabs for different upload types
    upload_tab1, upload_tab2, upload_tab3 = st.tabs(["📄 PDF Upload", "🎥 YouTube URL", "📊 CSV Management"])
    
    def add_document_to_csv(title, chunks, author, doc_type, genre, difficulty, source_type, tags, summary="", topic=""):
        """Add a new document entry to the CSV file."""
        try:
            import pandas as pd
            csv_file = "documents_metadata.csv"
            
            # Create new row
            new_row = {
                "title": title,
                "chunks": chunks,
                "author": author,
                "summary": summary,
                "type": doc_type,
                "genre": genre,
                "topic": topic,
                "difficulty": difficulty,
                "source_type": source_type,
                "tags": tags
            }
            
            # Read existing CSV or create new DataFrame
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
            else:
                df = pd.DataFrame(columns=["title", "chunks", "author", "summary", "type", "genre", "topic", "difficulty", "source_type", "tags"])
            
            # Add new row
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save back to CSV
            df.to_csv(csv_file, index=False)
            return True
        except Exception as e:
            st.error(f"Error updating CSV: {e}")
            return False

    def get_document_chunk_count(title):
        """Get the number of chunks for a document in Supabase."""
        try:
            result = supabase.table("documents_enhanced").select("id").eq("title", title).execute()
            return len(result.data) if result.data else 0
        except Exception as e:
            st.error(f"Error getting chunk count: {e}")
            return 0

    def delete_document_from_supabase(title):
        """Delete all chunks of a document from Supabase."""
        try:
            # First, check how many chunks exist
            chunk_count = get_document_chunk_count(title)
            
            if chunk_count > 0:
                # Delete all chunks with this title
                result = supabase.table("documents_enhanced").delete().eq("title", title).execute()
                return True, chunk_count
            else:
                return True, 0  # No chunks to delete, but operation successful
        except Exception as e:
            st.error(f"Error deleting from Supabase: {e}")
            return False, 0

    def update_document_metadata_in_supabase(old_title, new_metadata):
        """Update all chunks of a document in Supabase with new metadata."""
        try:
            # First, check how many chunks exist
            chunk_count = get_document_chunk_count(old_title)
            
            if chunk_count > 0:
                # Prepare update data for dedicated columns
                update_data = {
                    "title": new_metadata["title"],
                    "author": new_metadata["author"],
                    "doc_type": new_metadata["type"],
                    "genre": new_metadata["genre"],
                    "topic": new_metadata["topic"],
                    "difficulty": new_metadata["difficulty"],
                    "tags": new_metadata["tags"],
                    "source_type": new_metadata["source_type"],
                    "summary": new_metadata["summary"]
                }
                
                # Prepare JSON metadata for compatibility
                json_metadata = {
                    "title": new_metadata["title"],
                    "author": new_metadata["author"],
                    "type": new_metadata["type"],
                    "genre": new_metadata["genre"],
                    "topic": new_metadata["topic"],
                    "difficulty": new_metadata["difficulty"],
                    "tags": new_metadata["tags"],
                    "source_type": new_metadata["source_type"],
                    "summary": new_metadata["summary"]
                }
                
                # Update dedicated columns
                result1 = supabase.table("documents_enhanced").update(update_data).eq("title", old_title).execute()
                
                # Update JSON metadata (use new title in case title changed)
                result2 = supabase.table("documents_enhanced").update({"metadata": json_metadata}).eq("title", new_metadata["title"]).execute()
                
                return True, chunk_count
            else:
                return True, 0  # No chunks to update, but operation successful
        except Exception as e:
            st.error(f"Error updating Supabase metadata: {e}")
            return False, 0

    def preview_pdf_chunks(file, chunk_size, chunk_overlap, splitter_type):
        """Preview the first and last chunks of a PDF document."""
        try:
            # Reset file pointer
            file.seek(0)
            
            # Extract text from PDF
            from pypdf import PdfReader
            import io
            from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
            from langchain_core.documents import Document
            
            pdf_reader = PdfReader(io.BytesIO(file.read()))
            full_text = ""
            
            # Extract all text
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                full_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            if not full_text.strip():
                return None, "Could not extract text from PDF"
            
            # Choose text splitter based on user selection
            if splitter_type == "Recursive Character":
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                )
            else:  # Character splitter
                text_splitter = CharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separator="\n\n"  # Split on double newlines
                )
            
            # Create document object
            doc = Document(page_content=full_text, metadata={})
            
            # Split into chunks
            chunks = text_splitter.split_documents([doc])
            
            if not chunks:
                return None, "No chunks were created"
            
            return chunks, None
            
        except Exception as e:
            return None, f"Error processing document: {e}"

    def process_pdf_document(file, title, author, summary, doc_type, genre, topic, tags, difficulty, chunk_size, chunk_overlap, splitter_type):
        """Process PDF document: extract text, chunk, embed, and save to both CSV and Supabase."""
        try:
            # Reset file pointer
            file.seek(0)
            
            # Extract text from PDF
            from pypdf import PdfReader
            import io
            from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
            from langchain_core.documents import Document
            
            pdf_reader = PdfReader(io.BytesIO(file.read()))
            full_text = ""
            
            # Extract all text
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                full_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            if not full_text.strip():
                st.error("Could not extract text from PDF")
                return False
            
            # Create document metadata with proper field mapping
            metadata = {
                "title": title,
                "author": author,
                "summary": summary,
                "type": doc_type,  # This will be mapped to "doc_type" in enhanced store
                "genre": genre,
                "topic": topic,
                "tags": tags,
                "difficulty": difficulty,
                "source_type": "PDF",
                "source": f"PDF: {file.name}"
            }
            
            print(f"📊 Debug - Creating document with metadata: {metadata}")
            
            # Choose text splitter based on user selection
            if splitter_type == "Recursive Character":
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                )
            else:  # Character splitter
                text_splitter = CharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separator="\n\n"  # Split on double newlines
                )
            
            # Create document object
            doc = Document(page_content=full_text, metadata=metadata)
            
            # Split into chunks
            chunks = text_splitter.split_documents([doc])
            
            print(f"📊 Debug - Created {len(chunks)} chunks, first chunk metadata: {chunks[0].metadata if chunks else 'No chunks'}")
            
            # Add to vector store (Supabase)
            with st.spinner(f"Embedding {len(chunks)} chunks..."):
                vector_store.add_documents(chunks)
            
            # Get file content for tracker (use session state if available)
            file_key = f"file_content_{file.name.replace('.', '_')}"
            if file_key in st.session_state:
                file_content = st.session_state[file_key]
            else:
                file.seek(0)
                file_content = file.read()
            
            # Add to document tracker CSV
            doc_id = document_tracker.add_document_record(
                title=title,
                author=author,
                summary=summary,
                doc_type=doc_type,
                genre=genre,
                topic=topic,
                difficulty=difficulty,
                source_type="PDF",
                tags=tags,
                chunks=len(chunks),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                file_content=file_content,
                file_name=file.name,
                file_size=file.size
            )
            
            # Also update the legacy CSV for backward compatibility
            success = add_document_to_csv(
                title=title,
                chunks=len(chunks),
                author=author,
                doc_type=doc_type,
                genre=genre,
                difficulty=difficulty,
                source_type="PDF",
                tags=tags,
                summary=summary,
                topic=topic
            )
            
            if doc_id and success:
                st.success(f"✅ Added {len(chunks)} chunks to vector store and updated tracking systems")
                st.info(f"📄 A new row has been added to the CSV file: documents_metadata.csv")
                return True
            else:
                st.error("Failed to update tracking systems")
                return False
                
        except Exception as e:
            st.error(f"Error processing document: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return False

    with upload_tab1:
        st.subheader("Upload PDF Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type="pdf", 
            accept_multiple_files=True,
            help="Upload one or more PDF files to add to the vector store"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for i, file in enumerate(uploaded_files):
                st.write(f"- {file.name} ({file.size} bytes)")
                
                # Create a unique key for each file
                file_key = f"file_{i}_{file.name.replace('.', '_')}"
                
                # Store file content in session state to avoid multiple reads
                if f"file_content_{file_key}" not in st.session_state:
                    file_content = file.read()
                    file.seek(0)  # Reset file pointer
                    st.session_state[f"file_content_{file_key}"] = file_content
                else:
                    file_content = st.session_state[f"file_content_{file_key}"]
                
                # Check for duplicates only if not already checked
                if f"duplicate_checked_{file_key}" not in st.session_state:
                    is_duplicate, existing_record = document_tracker.is_duplicate_file(file_content, file.name)
                    st.session_state[f"duplicate_checked_{file_key}"] = True
                    st.session_state[f"is_duplicate_{file_key}"] = is_duplicate
                    st.session_state[f"existing_record_{file_key}"] = existing_record
                else:
                    is_duplicate = st.session_state[f"is_duplicate_{file_key}"]
                    existing_record = st.session_state[f"existing_record_{file_key}"]
                
                if is_duplicate and f"override_{file_key}" not in st.session_state:
                    st.error(f"🚫 Duplicate file detected: {file.name}")
                    st.info(f"This file was already uploaded on {existing_record.get('upload_date', 'Unknown date')}")
                    
                    with st.expander("📋 Existing Document Details", expanded=False):
                        st.json({
                            "Title": existing_record.get('title', 'Unknown'),
                            "Author": existing_record.get('author', 'Unknown'),
                            "Type": existing_record.get('type', 'Unknown'),
                            "Chunks": existing_record.get('chunks', 0),
                            "Upload Date": existing_record.get('upload_date', 'Unknown')
                        })
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🔄 Upload Anyway (Override)", key=f"override_{file_key}"):
                            st.session_state[f"override_{file_key}"] = True
                            st.warning("Proceeding with duplicate upload...")
                            st.rerun()
                    with col2:
                        if st.button(f"🗑️ Clear Duplicate Check", key=f"clear_dup_{file_key}"):
                            # Clear all duplicate-related session state for this file
                            keys_to_clear = [key for key in st.session_state.keys() if file_key in key or file.name.replace('.', '_') in key]
                            for key in keys_to_clear:
                                del st.session_state[key]
                            st.info("Duplicate check cleared. Please try again.")
                            st.rerun()
                    
                    if not st.session_state.get(f"override_{file_key}", False):
                        continue  # Skip this file
                
                # Show document title and generate metadata button
                if f"title_{file_key}" not in st.session_state:
                    # Pre-load title from filename
                    clean_title = file.name.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
                    st.session_state[f"title_{file_key}"] = clean_title
                
                # Display current title
                st.text_input(
                    f"Document Title for {file.name}", 
                    value=st.session_state[f"title_{file_key}"],
                    key=f"title_input_{file_key}",
                    on_change=lambda key=file_key: setattr(st.session_state, f"title_{key}", st.session_state[f"title_input_{key}"])
                )
                
                # Generate metadata button for each file
                if st.button(f"🤖 Generate Metadata", key=f"generate_{file_key}"):
                    with st.spinner(f"Analyzing {file.name} with GPT-4o-mini..."):
                        try:
                            # Extract text from PDF for analysis
                            from pypdf import PdfReader
                            import io
                            
                            # Read PDF content
                            pdf_reader = PdfReader(io.BytesIO(file.read()))
                            text_content = ""
                            
                            # Extract text from first few pages for analysis
                            max_pages = min(3, len(pdf_reader.pages))
                            for page_num in range(max_pages):
                                text_content += pdf_reader.pages[page_num].extract_text()
                            
                            # Limit text for API call (first 2000 characters)
                            analysis_text = text_content[:2000] if text_content else file.name
                            
                            # Generate metadata using GPT-4o-mini
                            metadata_prompt = f"""
                            Analyze this document and generate metadata. Document title: "{st.session_state[f'title_{file_key}']}"
                            
                            Document content preview:
                            {analysis_text}
                            
                            Please provide metadata in this exact JSON format:
                            {{
                                "title": "Clean, readable title",
                                "author": "Author name or 'Unknown'",
                                "summary": "2-3 sentence summary",
                                "type": "Book|Article|Report|Paper|Manual|Guide|Other",
                                "genre": "Fiction|Non-fiction|Technical|Academic|Business|Educational|Legal|Medical|Other",
                                "topic": "Main subject/topic area",
                                "tags": "comma, separated, relevant, tags",
                                "difficulty": "Beginner|Intermediate|Advanced|Expert"
                            }}
                            
                            Base your analysis on the content, filename, and document structure. Be concise and accurate.
                            """
                            
                            # Call GPT-4o-mini
                            response = llm.invoke(metadata_prompt)
                            
                            # Parse JSON response
                            import json
                            try:
                                # Extract JSON from response
                                response_text = response.content if hasattr(response, 'content') else str(response)
                                
                                # Find JSON in response
                                start_idx = response_text.find('{')
                                end_idx = response_text.rfind('}') + 1
                                
                                if start_idx != -1 and end_idx != -1:
                                    json_str = response_text[start_idx:end_idx]
                                    generated_metadata = json.loads(json_str)
                                else:
                                    raise ValueError("No JSON found in response")
                                
                                # Store in session state for this file
                                st.session_state[f"metadata_{file_key}"] = generated_metadata
                                st.success(f"✅ Metadata generated for {file.name}")
                                st.rerun()
                                
                            except Exception as parse_error:
                                st.error(f"Error parsing metadata: {parse_error}")
                                # Fallback metadata
                                st.session_state[f"metadata_{file_key}"] = {
                                    "title": st.session_state[f"title_{file_key}"],
                                    "author": "Unknown",
                                    "summary": "Document summary not available",
                                    "type": "Document",
                                    "genre": "Other",
                                    "topic": "General",
                                    "tags": "document, pdf",
                                    "difficulty": "Intermediate"
                                }
                                st.warning("Using fallback metadata. Please review and edit as needed.")
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"Error analyzing document: {e}")
                            continue
                
                # Show metadata form if generated
                if f"metadata_{file_key}" in st.session_state:
                    metadata = st.session_state[f"metadata_{file_key}"]
                    
                    with st.expander(f"📝 Review & Edit Metadata for {file.name}", expanded=True):
                        with st.form(f"metadata_form_{file_key}"):
                            st.subheader("Generated Metadata - Please Review & Edit")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                title = st.text_input("Title", value=metadata.get("title", ""), key=f"form_title_{file_key}")
                                author = st.text_input("Author", value=metadata.get("author", ""), key=f"author_{file_key}")
                                doc_type = st.text_input("Type", value=metadata.get("type", ""), key=f"type_{file_key}", 
                                    help="e.g., Book, Article, Report, Paper, Manual, Guide, etc.")
                                genre = st.text_input("Genre", value=metadata.get("genre", ""), key=f"genre_{file_key}",
                                    help="e.g., Fiction, Non-fiction, Technical, Academic, Business, Educational, etc.")
                            
                            with col2:
                                topic = st.text_input("Topic", value=metadata.get("topic", ""), key=f"topic_{file_key}")
                                difficulty = st.selectbox("Difficulty", 
                                    ["Beginner", "Intermediate", "Advanced", "Expert"],
                                    index=["Beginner", "Intermediate", "Advanced", "Expert"].index(metadata.get("difficulty", "Intermediate")) if metadata.get("difficulty") in ["Beginner", "Intermediate", "Advanced", "Expert"] else 1,
                                    key=f"difficulty_{file_key}")
                                tags = st.text_input("Tags", value=metadata.get("tags", ""), key=f"tags_{file_key}")
                            
                            summary = st.text_area("Summary", value=metadata.get("summary", ""), height=100, key=f"summary_{file_key}")
                            
                            # Chunking parameters
                            st.subheader("Chunking Parameters")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                chunk_size = st.slider("Chunk Size", 500, 5000, 1000, 100, key=f"chunk_size_{file_key}")
                            with col2:
                                chunk_overlap = st.slider("Chunk Overlap", 50, 500, 100, 25, key=f"chunk_overlap_{file_key}")
                            with col3:
                                splitter_type = st.selectbox("Text Splitter", 
                                    ["Recursive Character", "Character"],
                                    index=0,
                                    key=f"splitter_{file_key}",
                                    help="Recursive Character: Better for most documents. Character: Splits on specific separators.")
                            
                            # Preview chunks button
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.form_submit_button("🔍 Preview Chunks", type="secondary"):
                                    with st.spinner("Generating chunk preview..."):
                                        chunks, error = preview_pdf_chunks(file, chunk_size, chunk_overlap, splitter_type)
                                        if error:
                                            st.error(f"Error: {error}")
                                        else:
                                            # Store chunks in session state for preview
                                            st.session_state[f"chunks_preview_{file_key}"] = chunks
                                            st.success(f"✅ Generated {len(chunks)} chunks for preview")
                                            st.rerun()
                            
                            with col2:
                                # Only show upload button if chunks have been previewed
                                if f"chunks_preview_{file_key}" in st.session_state:
                                    if st.form_submit_button("✅ Upload to Supabase", type="primary"):
                                        # Process the document
                                        success = process_pdf_document(
                                            file, title, author, summary, doc_type, genre, topic, tags, difficulty, chunk_size, chunk_overlap, splitter_type
                                        )
                                        if success:
                                            st.success(f"🎉 Successfully processed {title}!")
                                            # Clear all session state related to this file to prevent duplicate detection issues
                                            keys_to_remove = [key for key in st.session_state.keys() if file_key in key]
                                            for key in keys_to_remove:
                                                del st.session_state[key]
                                            
                                            # Also clear any duplicate-related session state
                                            duplicate_keys = [key for key in st.session_state.keys() if 'duplicate' in key and file.name.replace('.', '_') in key]
                                            for key in duplicate_keys:
                                                del st.session_state[key]
                                            
                                            st.info("✅ File uploaded successfully. You can now upload the same file again if needed.")
                                            st.rerun()
                                else:
                                    st.form_submit_button("✅ Upload to Supabase", type="primary", disabled=True, help="Preview chunks first")
                            
                            with col3:
                                if st.form_submit_button("🗑️ Cancel", type="secondary"):
                                    # Clear the metadata from session state
                                    for key in list(st.session_state.keys()):
                                        if file_key in key:
                                            del st.session_state[key]
                                    st.rerun()
                        
                        # Show chunk preview if available
                        if f"chunks_preview_{file_key}" in st.session_state:
                            chunks = st.session_state[f"chunks_preview_{file_key}"]
                            
                            st.subheader(f"📄 Chunk Preview ({len(chunks)} total chunks)")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**🟢 First Chunk:**")
                                st.text_area(
                                    "First chunk content:",
                                    value=chunks[0].page_content,
                                    height=200,
                                    key=f"first_chunk_{file_key}",
                                    disabled=True
                                )
                                st.caption(f"Length: {len(chunks[0].page_content)} characters")
                            
                            with col2:
                                st.markdown("**🔴 Last Chunk:**")
                                st.text_area(
                                    "Last chunk content:",
                                    value=chunks[-1].page_content,
                                    height=200,
                                    key=f"last_chunk_{file_key}",
                                    disabled=True
                                )
                                st.caption(f"Length: {len(chunks[-1].page_content)} characters")
                            
                            # Show chunk statistics
                            chunk_lengths = [len(chunk.page_content) for chunk in chunks]
                            avg_length = sum(chunk_lengths) / len(chunk_lengths)
                            min_length = min(chunk_lengths)
                            max_length = max(chunk_lengths)
                            
                            st.info(f"""
                            **📊 Chunk Statistics:**
                            - Total chunks: {len(chunks)}
                            - Average length: {avg_length:.0f} characters
                            - Shortest chunk: {min_length} characters
                            - Longest chunk: {max_length} characters
                            """)
                            
                            if len(chunks) > 2:
                                st.markdown("**🔍 All Chunk Lengths:**")
                                import pandas as pd
                                chunk_data = []
                                for i, chunk in enumerate(chunks):
                                    chunk_data.append({
                                        "Chunk #": i + 1,
                                        "Length": len(chunk.page_content),
                                        "Preview": chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
                                    })
                                
                                df_chunks = pd.DataFrame(chunk_data)
                                st.dataframe(df_chunks, use_container_width=True, height=200)
    
    with upload_tab2:
        st.subheader("Add YouTube Videos")
        youtube_url = st.text_input(
            "YouTube URL", 
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter a YouTube URL to extract transcript and add to vector store"
        )
        
        if youtube_url:
            # Extract video ID
            video_id = extract_video_id(youtube_url)
            
            if not video_id:
                st.error("Invalid YouTube URL. Please enter a valid YouTube URL.")
            else:
                st.success(f"✅ Valid YouTube URL detected. Video ID: {video_id}")
                
                # Check for duplicates first
                is_duplicate, existing_record = document_tracker.is_duplicate_url(youtube_url, video_id)
                
                if is_duplicate:
                    st.error(f"🚫 Duplicate YouTube video detected: {video_id}")
                    st.info(f"This video was already uploaded on {existing_record.get('upload_date', 'Unknown date')}")
                    
                    with st.expander("📋 Existing Video Details", expanded=False):
                        st.json({
                            "Title": existing_record.get('title', 'Unknown'),
                            "Author": existing_record.get('author', 'Unknown'),
                            "Type": existing_record.get('type', 'Unknown'),
                            "Chunks": existing_record.get('chunks', 0),
                            "Upload Date": existing_record.get('upload_date', 'Unknown'),
                            "Video ID": existing_record.get('video_id', 'Unknown')
                        })
                    
                    if st.button(f"🔄 Upload Anyway (Override)", key=f"override_youtube_{video_id}"):
                        st.warning("Proceeding with duplicate upload...")
                        # Continue with normal upload process
                    else:
                        st.stop()  # Stop processing this video
                
                # Show video title input
                if f"youtube_title_{video_id}" not in st.session_state:
                    st.session_state[f"youtube_title_{video_id}"] = f"YouTube Video {video_id}"
                
                st.text_input(
                    "Video Title", 
                    value=st.session_state[f"youtube_title_{video_id}"],
                    key=f"youtube_title_input_{video_id}",
                    on_change=lambda vid=video_id: setattr(st.session_state, f"youtube_title_{vid}", st.session_state[f"youtube_title_input_{vid}"])
                )
                
                # Generate metadata button
                if st.button(f"🤖 Generate Metadata", key=f"generate_youtube_{video_id}"):
                    with st.spinner("Fetching video metadata and transcript..."):
                        try:
                            # Get video metadata
                            video_metadata = get_youtube_metadata(video_id)
                            
                            # Get transcript from Supadata API
                            transcript = get_youtube_transcript(video_id)
                            
                            if not transcript:
                                st.error("Could not retrieve transcript for this video. It may not have captions available.")
                            else:
                                # Clean transcript using GPT-4o-mini
                                cleaned_transcript = clean_youtube_transcript(transcript)
                                
                                # Generate enhanced metadata using GPT-4o-mini
                                enhanced_metadata = generate_youtube_metadata(
                                    st.session_state[f"youtube_title_{video_id}"], 
                                    cleaned_transcript, 
                                    video_metadata
                                )
                                
                                # Store in session state
                                st.session_state[f"youtube_metadata_{video_id}"] = enhanced_metadata
                                st.session_state[f"youtube_transcript_{video_id}"] = cleaned_transcript
                                
                                st.success("✅ Metadata and transcript generated successfully!")
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"Error processing YouTube video: {e}")
                
                # Show metadata form if generated
                if f"youtube_metadata_{video_id}" in st.session_state:
                    metadata = st.session_state[f"youtube_metadata_{video_id}"]
                    
                    with st.expander("📝 Review & Edit YouTube Video Metadata", expanded=True):
                        with st.form(f"youtube_metadata_form_{video_id}"):
                            st.subheader("Generated Metadata - Please Review & Edit")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                title = st.text_input("Title", value=metadata.get("title", ""), key=f"youtube_form_title_{video_id}")
                                author = st.text_input("Author/Speaker", value=metadata.get("author", ""), key=f"youtube_author_{video_id}")
                                youtube_channel = st.text_input("YouTube Channel", value=metadata.get("youtube_channel", ""), key=f"youtube_channel_{video_id}", disabled=True)
                                genre = st.selectbox("Genre", 
                                    ["Educational", "Tutorial", "Documentary", "Interview", "Lecture", "Entertainment", "News", "Other"],
                                    index=["Educational", "Tutorial", "Documentary", "Interview", "Lecture", "Entertainment", "News", "Other"].index(metadata.get("genre", "Educational")) if metadata.get("genre") in ["Educational", "Tutorial", "Documentary", "Interview", "Lecture", "Entertainment", "News", "Other"] else 0,
                                    key=f"youtube_genre_{video_id}")
                            
                            with col2:
                                topic = st.text_input("Topic", value=metadata.get("topic", ""), key=f"youtube_topic_{video_id}")
                                difficulty = st.selectbox("Difficulty", 
                                    ["Beginner", "Intermediate", "Advanced", "Expert"],
                                    index=["Beginner", "Intermediate", "Advanced", "Expert"].index(metadata.get("difficulty", "Intermediate")) if metadata.get("difficulty") in ["Beginner", "Intermediate", "Advanced", "Expert"] else 1,
                                    key=f"youtube_difficulty_{video_id}")
                                tags = st.text_input("Tags", value=metadata.get("tags", ""), key=f"youtube_tags_{video_id}")
                            
                            summary = st.text_area("Summary", value=metadata.get("summary", ""), height=100, key=f"youtube_summary_{video_id}")
                            
                            # Chunking parameters for YouTube
                            st.subheader("Chunking Parameters")
                            col1, col2 = st.columns(2)
                            with col1:
                                chunk_size = st.slider("Chunk Size", 200, 2000, 400, 50, key=f"youtube_chunk_size_{video_id}")
                            with col2:
                                chunk_overlap = st.slider("Chunk Overlap", 0, 300, 200, 25, key=f"youtube_chunk_overlap_{video_id}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.form_submit_button("✅ Upload to Supabase", type="primary"):
                                    # Process the YouTube video
                                    success = process_youtube_video(
                                        video_id,
                                        st.session_state[f"youtube_transcript_{video_id}"],
                                        title, author, summary, genre, topic, tags, difficulty,
                                        metadata, chunk_size, chunk_overlap
                                    )
                                    if success:
                                        st.success(f"🎉 Successfully processed {title}!")
                                        # Clear all session state related to this file
                                        keys_to_remove = [key for key in st.session_state.keys() if f"youtube_{video_id}" in key]
                                        for key in keys_to_remove:
                                            del st.session_state[key]
                                        st.rerun()
                            
                            with col2:
                                if st.form_submit_button("🗑️ Cancel", type="secondary"):
                                    # Clear the metadata from session state
                                    for key in list(st.session_state.keys()):
                                        if video_id in key:
                                            del st.session_state[key]
                                    st.rerun()

    with upload_tab3:
        st.subheader("📊 CSV Management")
        st.write("Manage the document metadata CSV file for faster loading.")
        
        # Document Tracker Statistics
        st.subheader("📈 Document Tracker Statistics")
        tracker_stats = document_tracker.get_document_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", tracker_stats.get("total", 0))
        with col2:
            st.metric("Total Chunks", tracker_stats.get("total_chunks", 0))
        with col3:
            st.metric("Document Types", len(tracker_stats.get("by_type", {})))
        with col4:
            st.metric("Source Types", len(tracker_stats.get("by_source", {})))
        
        # Show breakdown by type and source
        if tracker_stats.get("by_type"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**By Document Type:**")
                for doc_type, count in tracker_stats["by_type"].items():
                    st.write(f"- {doc_type}: {count}")
            
            with col2:
                st.write("**By Source Type:**")
                for source_type, count in tracker_stats["by_source"].items():
                    st.write(f"- {source_type}: {count}")
        
        st.divider()
        
        # Document Tracker Management
        st.subheader("🔍 Document Search & Management")
        
        col1, col2 = st.columns(2)
        with col1:
            search_query = st.text_input("Search Documents", placeholder="Search by title, author, tags, or topic...")
        with col2:
            if st.button("🔍 Search"):
                if search_query:
                    search_results = document_tracker.search_documents(search_query)
                    if not search_results.empty:
                        st.success(f"Found {len(search_results)} matching documents")
                        st.dataframe(search_results[['title', 'author', 'type', 'chunks', 'upload_date']], use_container_width=True)
                    else:
                        st.info("No documents found matching your search")
        
        # View all tracked documents
        if st.button("📋 View All Tracked Documents"):
            all_docs = document_tracker.get_all_documents()
            if not all_docs.empty:
                st.dataframe(all_docs, use_container_width=True)
            else:
                st.info("No documents found in tracker")
        
        # CSV Editing Interface
        st.subheader("✏️ Edit Document Metadata")
        
        # Load current documents for editing
        all_docs = document_tracker.get_all_documents()
        
        if not all_docs.empty:
            # Select document to edit
            doc_titles = all_docs['title'].tolist()
            selected_title = st.selectbox("Select Document to Edit", [""] + doc_titles)
            
            if selected_title:
                # Get the selected document
                selected_doc = all_docs[all_docs['title'] == selected_title].iloc[0]
                
                # Show chunk information
                csv_chunks = selected_doc.get('chunks', 0)
                supabase_chunks = get_document_chunk_count(selected_title)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CSV Chunks", csv_chunks)
                with col2:
                    st.metric("Supabase Chunks", supabase_chunks)
                with col3:
                    if csv_chunks == supabase_chunks:
                        st.success("✅ Synced")
                    else:
                        st.warning("⚠️ Out of sync")
                
                with st.form("edit_document_form"):
                    st.subheader(f"Editing: {selected_title}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        new_title = st.text_input("Title", value=selected_doc.get('title', ''))
                        new_author = st.text_input("Author", value=selected_doc.get('author', ''))
                        new_type = st.selectbox("Type", 
                            ["Book", "Article", "Report", "Paper", "Manual", "Guide", "Video", "Other"],
                            index=["Book", "Article", "Report", "Paper", "Manual", "Guide", "Video", "Other"].index(selected_doc.get('type', 'Other')) if selected_doc.get('type') in ["Book", "Article", "Report", "Paper", "Manual", "Guide", "Video", "Other"] else 7)
                        new_genre = st.selectbox("Genre", 
                            ["Fiction", "Non-fiction", "Technical", "Academic", "Business", "Educational", "Legal", "Medical", "Other"],
                            index=["Fiction", "Non-fiction", "Technical", "Academic", "Business", "Educational", "Legal", "Medical", "Other"].index(selected_doc.get('genre', 'Other')) if selected_doc.get('genre') in ["Fiction", "Non-fiction", "Technical", "Academic", "Business", "Educational", "Legal", "Medical", "Other"] else 8)
                    
                    with col2:
                        new_topic = st.text_input("Topic", value=selected_doc.get('topic', ''))
                        new_difficulty = st.selectbox("Difficulty", 
                            ["Beginner", "Intermediate", "Advanced", "Expert"],
                            index=["Beginner", "Intermediate", "Advanced", "Expert"].index(selected_doc.get('difficulty', 'Intermediate')) if selected_doc.get('difficulty') in ["Beginner", "Intermediate", "Advanced", "Expert"] else 1)
                        new_tags = st.text_input("Tags", value=selected_doc.get('tags', ''))
                        new_source_type = st.selectbox("Source Type",
                            ["PDF", "YouTube", "Web", "Other"],
                            index=["PDF", "YouTube", "Web", "Other"].index(selected_doc.get('source_type', 'Other')) if selected_doc.get('source_type') in ["PDF", "YouTube", "Web", "Other"] else 3)
                    
                    new_summary = st.text_area("Summary", value=selected_doc.get('summary', ''), height=100)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.form_submit_button("💾 Save Changes", type="primary"):
                            try:
                                # Update the document in the tracker CSV
                                all_docs.loc[all_docs['title'] == selected_title, 'title'] = new_title
                                all_docs.loc[all_docs['title'] == selected_title, 'author'] = new_author
                                all_docs.loc[all_docs['title'] == selected_title, 'type'] = new_type
                                all_docs.loc[all_docs['title'] == selected_title, 'genre'] = new_genre
                                all_docs.loc[all_docs['title'] == selected_title, 'topic'] = new_topic
                                all_docs.loc[all_docs['title'] == selected_title, 'difficulty'] = new_difficulty
                                all_docs.loc[all_docs['title'] == selected_title, 'tags'] = new_tags
                                all_docs.loc[all_docs['title'] == selected_title, 'source_type'] = new_source_type
                                all_docs.loc[all_docs['title'] == selected_title, 'summary'] = new_summary
                                
                                # Save back to CSV
                                all_docs.to_csv(document_tracker.csv_file, index=False)
                                
                                # Also update the enhanced Supabase table
                                try:
                                    # Prepare new metadata
                                    new_metadata = {
                                        "title": new_title,
                                        "author": new_author,
                                        "type": new_type,
                                        "genre": new_genre,
                                        "topic": new_topic,
                                        "difficulty": new_difficulty,
                                        "tags": new_tags,
                                        "source_type": new_source_type,
                                        "summary": new_summary
                                    }
                                    
                                    # Update all chunks in Supabase
                                    supabase_success, chunk_count = update_document_metadata_in_supabase(selected_title, new_metadata)
                                    
                                    if supabase_success:
                                        if chunk_count > 0:
                                            st.success(f"✅ Document updated in CSV and {chunk_count} chunks updated in Supabase!")
                                            
                                            # If title changed, show additional info
                                            if selected_title != new_title:
                                                st.info(f"📝 Title changed from '{selected_title}' to '{new_title}' across all chunks")
                                        else:
                                            st.success("✅ Document updated in CSV! (No chunks found in Supabase to update)")
                                    else:
                                        st.warning("⚠️ CSV updated but Supabase update failed")
                                        st.info("Use 'Sync to Supabase' button to retry Supabase update.")
                                    
                                except Exception as e:
                                    st.warning(f"⚠️ CSV updated but Supabase update failed: {e}")
                                    st.info("Document metadata updated in CSV. Use 'Sync to Supabase' button to retry Supabase update.")
                                
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error updating document: {e}")
                    
                    with col2:
                        if st.form_submit_button("🗑️ Delete Document", type="secondary"):
                            try:
                                # Remove from CSV
                                doc_id = selected_doc.get('id', '')
                                success = document_tracker.remove_document(doc_id)
                                
                                if success:
                                    # Also remove ALL chunks from Supabase enhanced store
                                    supabase_success, chunk_count = delete_document_from_supabase(selected_title)
                                    
                                    if supabase_success:
                                        if chunk_count > 0:
                                            st.success(f"✅ Document deleted from CSV and {chunk_count} chunks deleted from Supabase!")
                                        else:
                                            st.success("✅ Document deleted from CSV! (No chunks found in Supabase)")
                                    else:
                                        st.warning(f"⚠️ Deleted from CSV but Supabase deletion failed")
                                        st.info("💡 You may need to manually delete the chunks from Supabase")
                                    
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to delete document from CSV")
                                    
                            except Exception as e:
                                st.error(f"❌ Error deleting document: {e}")
                    
                    with col3:
                        if st.form_submit_button("🔄 Sync to Supabase", type="secondary"):
                            try:
                                # Prepare metadata for sync
                                sync_metadata = {
                                    "title": new_title,
                                    "author": new_author,
                                    "type": new_type,
                                    "genre": new_genre,
                                    "topic": new_topic,
                                    "difficulty": new_difficulty,
                                    "tags": new_tags,
                                    "source_type": new_source_type,
                                    "summary": new_summary
                                }
                                
                                # Use helper function to update metadata
                                supabase_success, chunk_count = update_document_metadata_in_supabase(selected_title, sync_metadata)
                                
                                if supabase_success:
                                    if chunk_count > 0:
                                        st.success(f"✅ {chunk_count} chunks synced to Supabase!")
                                        
                                        # If title changed, show additional info
                                        if selected_title != new_title:
                                            st.info(f"📝 Title updated from '{selected_title}' to '{new_title}' across all chunks")
                                    else:
                                        st.warning("⚠️ No chunks found in Supabase for this document")
                                        st.info("💡 The document may not have been uploaded to Supabase yet")
                                else:
                                    st.error("❌ Sync failed")
                                    st.info("💡 Check your Supabase connection and table structure")
                                
                            except Exception as e:
                                st.error(f"❌ Sync failed: {e}")
                                st.info("💡 Check your Supabase connection and table structure")
        else:
            st.info("No documents available for editing. Upload some documents first!")
        
        st.divider()
        
        # Export Options
        st.subheader("📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Export Tracker to CSV"):
                success = document_tracker.export_metadata_csv("documents_metadata_export.csv")
                if success:
                    st.success("✅ Exported tracker data to documents_metadata_export.csv")
                    
                    # Offer download
                    try:
                        with open("documents_metadata_export.csv", "rb") as file:
                            st.download_button(
                                label="⬇️ Download Export",
                                data=file,
                                file_name="documents_metadata_export.csv",
                                mime="text/csv"
                            )
                    except Exception as e:
                        st.error(f"Error creating download: {e}")
        
        with col2:
            if st.button("📊 Export Full Tracker"):
                try:
                    all_docs = document_tracker.get_all_documents()
                    if not all_docs.empty:
                        all_docs.to_csv("full_document_tracker_export.csv", index=False)
                        st.success("✅ Exported full tracker to full_document_tracker_export.csv")
                        
                        # Offer download
                        with open("full_document_tracker_export.csv", "rb") as file:
                            st.download_button(
                                label="⬇️ Download Full Export",
                                data=file,
                                file_name="full_document_tracker_export.csv",
                                mime="text/csv"
                            )
                    else:
                        st.info("No documents to export")
                except Exception as e:
                    st.error(f"Error exporting full tracker: {e}")
        
        with col3:
            if st.button("🧹 Clean Duplicates"):
                try:
                    all_docs = document_tracker.get_all_documents()
                    if not all_docs.empty:
                        # Find duplicates by file hash
                        duplicates = all_docs[all_docs.duplicated(subset=['file_hash'], keep='first')]
                        
                        if not duplicates.empty:
                            st.warning(f"Found {len(duplicates)} potential duplicates")
                            st.dataframe(duplicates[['title', 'author', 'upload_date', 'file_hash']], use_container_width=True)
                            
                            if st.button("🗑️ Remove Duplicates", type="secondary"):
                                # Remove duplicates (keep first occurrence)
                                cleaned_df = all_docs.drop_duplicates(subset=['file_hash'], keep='first')
                                cleaned_df.to_csv(document_tracker.csv_file, index=False)
                                st.success(f"✅ Removed {len(duplicates)} duplicates")
                                st.rerun()
                        else:
                            st.success("✅ No duplicates found")
                    else:
                        st.info("No documents to check")
                except Exception as e:
                    st.error(f"Error checking duplicates: {e}")
        
        st.divider()
        
        # Legacy CSV Management
        st.subheader("📄 Legacy CSV Management")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📄 View Legacy CSV"):
                try:
                    import pandas as pd
                    if os.path.exists("documents_metadata.csv"):
                        df = pd.read_csv("documents_metadata.csv")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("Legacy CSV file not found")
                except Exception as e:
                    st.error(f"Error reading legacy CSV: {e}")

        with col2:
            if st.button("🔄 Sync from Supabase"):
                try:
                    with st.spinner("Syncing from Supabase..."):
                        table_name = "documents_enhanced"
                        
                        result = supabase.table(table_name).select("title, author, doc_type, genre, difficulty, source_type, tags, summary").execute()
                        
                        documents = result.data
                        
                        if documents:
                            # Count chunks by title for enhanced store
                            title_counts = {}
                            doc_by_title = {}
                            
                            for doc in documents:
                                title = doc.get("title", "Unknown")
                                if title not in title_counts:
                                    title_counts[title] = 0
                                    doc_by_title[title] = doc
                                title_counts[title] += 1
                            
                            # Create DataFrame
                            data = []
                            for title, count in title_counts.items():
                                doc = doc_by_title[title]
                                data.append({
                                    "title": title,
                                    "chunks": count,
                                    "author": doc.get("author", "Unknown"),
                                    "summary": doc.get("summary", ""),
                                    "type": doc.get("doc_type", "Unknown"),
                                    "genre": doc.get("genre", "Unknown"),
                                    "topic": doc.get("topic", "Unknown"),
                                    "difficulty": doc.get("difficulty", "Unknown"),
                                    "source_type": doc.get("source_type", "Unknown"),
                                    "tags": doc.get("tags", "Unknown")
                                })
                            
                            df = pd.DataFrame(data)
                            df.to_csv("documents_metadata.csv", index=False)
                            st.success(f"✅ Synced {len(df)} documents to legacy CSV")
                        else:
                            st.warning("No documents found in Supabase")
                except Exception as e:
                    st.error(f"Error syncing from Supabase: {e}")

        with col3:
            uploaded_csv = st.file_uploader("📤 Upload CSV", type="csv", help="Upload a CSV file to replace the current document registry")
            if uploaded_csv:
                try:
                    import pandas as pd
                    df = pd.read_csv(uploaded_csv)
                    df.to_csv("documents_metadata.csv", index=False)
                    st.success(f"✅ Uploaded CSV with {len(df)} documents")
                except Exception as e:
                    st.error(f"Error uploading CSV: {e}")

    # YouTube processing functions
    def extract_video_id(url):
        """Extract the video ID from a YouTube URL."""
        import re
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

    def get_youtube_metadata(video_id):
        """Get video metadata from YouTube."""
        try:
            import requests
            import re
            
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
                    
                # Extract channel name
                channel_match = re.search(r'<link itemprop="name" content="([^"]+)"', html_content)
                if channel_match:
                    metadata["youtube_channel"] = channel_match.group(1)
                else:
                    channel_match2 = re.search(r'"author":"([^"]+)"', html_content)
                    if channel_match2:
                        metadata["youtube_channel"] = channel_match2.group(1)
                    else:
                        metadata["youtube_channel"] = "Unknown Channel"
            else:
                metadata["title"] = f"YouTube Video {video_id}"
                metadata["youtube_channel"] = "Unknown Channel"
            
            return metadata
        except Exception as e:
            st.error(f"Error fetching video metadata: {e}")
            return {
                "title": f"YouTube Video {video_id}",
                "youtube_channel": "Unknown Channel",
                "video_id": video_id,
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
                "source_type": "youtube_video"
            }

    def get_youtube_transcript(video_id):
        """Get transcript using Supadata API."""
        try:
            import requests
            
            # Supadata API configuration
            SUPADATA_API_URL = "https://api.supadata.ai"
            SUPADATA_TRANSCRIPT_ENDPOINT = "/v1/youtube/transcript"
            SUPADATA_API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiIsImtpZCI6IjEifQ.eyJpc3MiOiJuYWRsZXMiLCJpYXQiOiIxNzQ3OTIwNTIyIiwicHVycG9zZSI6ImFwaV9hdXRoZW50aWNhdGlvbiIsInN1YiI6IjEwMjk3YjAyYThlZjRhOTdhNmFjNjUwNjYxYWVlZjNiIn0.5OwI0aFR_BfgrDp2c55muHS9OyVX6XxHHPhULTzqdRY"
            
            api_url = f"{SUPADATA_API_URL}{SUPADATA_TRANSCRIPT_ENDPOINT}"
            
            params = {"videoId": video_id}
            headers = {"X-API-Key": SUPADATA_API_KEY}
            
            response = requests.get(api_url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if "content" in data:
                    transcript = " ".join([segment.get("text", "") for segment in data.get("content", [])])
                    return transcript
                else:
                    return None
            else:
                st.error(f"Supadata API error: Status code {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Error getting transcript: {e}")
            return None

    def clean_youtube_transcript(transcript):
        """Clean and format transcript using GPT-4o-mini."""
        try:
            system_prompt = """You are an expert in grammar corrections and textual structuring.

Correct the classification of the provided text, adding commas, periods, question marks and other symbols necessary for natural and consistent reading. Do not change any words, just adjust the punctuation according to the grammatical rules and context.

Organize your content using markdown, structuring it with titles, subtitles, lists or other protected elements to clearly highlight the topics and information captured. Leave it in English and remember to always maintain the original formatting.

Textual organization should always be a priority according to the content of the text, as well as the appropriate title, which must make sense."""
            
            # Limit transcript length
            max_content_length = 12000
            if len(transcript) > max_content_length:
                transcript = transcript[:max_content_length] + "\n\n[Transcript truncated due to length]"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is a YouTube transcript that needs cleaning and formatting:\n\n{transcript}"}
            ]
            
            response = llm.invoke(messages)
            return response.content
            
        except Exception as e:
            st.error(f"Error cleaning transcript: {e}")
            return transcript

    def generate_youtube_metadata(title, transcript_sample, video_metadata):
        """Generate enhanced metadata for YouTube video."""
        try:
            system_message = """You are a metadata expert who creates high-quality content summaries and tags for YouTube videos.
Follow these instructions carefully:
1. Create a concise summary using clear, concise language with active voice
2. Identify the genre/topic and content type
3. Identify the ACTUAL AUTHOR of the content (not the YouTube channel) from the title and content
4. Assign a difficulty rating (Beginner, Intermediate, Advanced, Expert) based on complexity and target audience
5. Generate relevant tags that would be useful in a chatbot context

Format your response exactly as follows:
Summary: [Your summary here]
Genre: [Genre]
Topic: [Topic]
Type: [Content type - should be "Video"]
Author: [The actual author/speaker of the content, not the YouTube channel]
Tags: [tag1, tag2, tag3, etc.]
Difficulty: [Beginner/Intermediate/Advanced/Expert]"""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Generate metadata for YouTube video titled '{title}' with this transcript sample: {transcript_sample[:1500]}..."}
            ]
            
            response = llm.invoke(messages)
            response_text = response.content
            
            # Parse response
            import re
            metadata_dict = {}
            try:
                metadata_dict["summary"] = re.search(r"Summary: (.*?)(?:\n|$)", response_text, re.DOTALL).group(1).strip()
                metadata_dict["genre"] = re.search(r"Genre: (.*?)(?:\n|$)", response_text).group(1).strip()
                metadata_dict["topic"] = re.search(r"Topic: (.*?)(?:\n|$)", response_text).group(1).strip()
                metadata_dict["type"] = re.search(r"Type: (.*?)(?:\n|$)", response_text).group(1).strip()
                metadata_dict["author"] = re.search(r"Author: (.*?)(?:\n|$)", response_text).group(1).strip()
                metadata_dict["tags"] = re.search(r"Tags: (.*?)(?:\n|$)", response_text).group(1).strip()
                metadata_dict["difficulty"] = re.search(r"Difficulty: (.*?)(?:\n|$)", response_text).group(1).strip()
                
                # Add video metadata
                for key, value in video_metadata.items():
                    if key != "author":  # Don't overwrite author with channel name
                        metadata_dict[key] = value
                
                metadata_dict["youtube_channel"] = video_metadata.get("youtube_channel", "Unknown Channel")
                
            except (AttributeError, Exception) as e:
                st.error(f"Error parsing metadata: {e}")
                # Fallback metadata
                metadata_dict = {
                    "summary": "Summary extraction failed",
                    "genre": "Educational",
                    "topic": "Unknown",
                    "type": "Video",
                    "author": "Unknown",
                    "tags": "youtube, video",
                    "difficulty": "Intermediate"
                }
                metadata_dict.update(video_metadata)
            
            return metadata_dict
            
        except Exception as e:
            st.error(f"Error generating metadata: {e}")
            return video_metadata

    def process_youtube_video(video_id, transcript, title, author, summary, genre, topic, tags, difficulty, original_metadata, chunk_size, chunk_overlap):
        """Process YouTube video: chunk, embed, and save to both CSV and Supabase."""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_core.documents import Document
            
            # Create document metadata
            metadata = {
                "title": title,
                "author": author,
                "summary": summary,
                "type": "Video",
                "genre": genre,
                "topic": topic,
                "tags": tags,
                "difficulty": difficulty,
                "source_type": "YouTube",
                "video_id": video_id,
                "youtube_channel": original_metadata.get("youtube_channel", "Unknown"),
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
                "source": f"YouTube: {title}"
            }
            
            # Split text into chunks with user-specified parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            
            # Create document object
            doc = Document(page_content=transcript, metadata=metadata)
            
            # Split into chunks
            chunks = text_splitter.split_documents([doc])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
            
            # Add to vector store (Supabase)
            with st.spinner(f"Embedding {len(chunks)} chunks..."):
                vector_store.add_documents(chunks)
            
            # Add to document tracker CSV
            source_url = f"https://www.youtube.com/watch?v={video_id}"
            
            doc_id = document_tracker.add_document_record(
                title=title,
                author=author,
                summary=summary,
                doc_type="Video",
                genre=genre,
                topic=topic,
                difficulty=difficulty,
                source_type="YouTube",
                tags=tags,
                chunks=len(chunks),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_url=source_url,
                video_id=video_id
            )
            
            # Also update the legacy CSV for backward compatibility
            success = add_document_to_csv(
                title=title,
                chunks=len(chunks),
                author=author,
                doc_type="Video",
                genre=genre,
                difficulty=difficulty,
                source_type="YouTube",
                tags=tags,
                summary=summary,
                topic=topic
            )
            
            if doc_id and success:
                st.success(f"✅ Added {len(chunks)} chunks to vector store and updated tracking systems")
                return True
            else:
                st.error("Failed to update tracking systems")
                return False
                
        except Exception as e:
            st.error(f"Error processing YouTube video: {e}")
            return False

