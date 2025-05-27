# import basics
import os
import uuid
from dotenv import load_dotenv

# import streamlit
import streamlit as st

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

# load environment variables
load_dotenv()  
print("DEBUG - OpenAI API Key:", os.environ.get("OPENAI_API_KEY")[:10] + "..." if os.environ.get("OPENAI_API_KEY") else "NOT FOUND")

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

# initiating vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)
 
# initiating llm
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

# pulling prompt from hub
prompt = hub.pull("hwchase17/openai-functions-agent")


# creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# initiating streamlit app
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
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
        st.markdown("### Tips")
        st.markdown("• Be specific in your questions")
        st.markdown("• Provide feedback to improve responses")
        st.markdown("• Check the Vector Store tab to see available documents")
        
        st.divider()
        st.markdown("### Debug")
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
                    "Type": row.get("type", "Unknown"),
                    "Genre": row.get("genre", "Unknown"),
                    "Difficulty": row.get("difficulty", "Unknown"),
                    "Source Type": row.get("source_type", "Unknown"),
                    "Tags": str(row.get("tags", "Unknown"))[:50] + "..." if len(str(row.get("tags", "Unknown"))) > 50 else str(row.get("tags", "Unknown"))
                })
            
            total_chunks = df["chunks"].sum() if "chunks" in df.columns else len(df)
            st.success(f"✅ Loaded {len(df)} documents with {total_chunks} total chunks from CSV")
            
        else:
            st.warning("📄 CSV file not found. Falling back to Supabase query...")
            # Fallback to Supabase if CSV doesn't exist
            with st.spinner("Loading from Supabase..."):
                result = supabase.table("documents").select("id, metadata").limit(50000).execute()
                documents = result.data
            
            if not documents:
                st.info("No documents found in the vector store.")
                display_data = []
                total_chunks = 0
            else:
                # Count documents by title and collect metadata
                title_counts = {}
                metadata_by_title = {}
                
                for doc in documents:
                    metadata = doc.get("metadata", {})
                    title = metadata.get("title", "Unknown")
                    
                    if title not in title_counts:
                        title_counts[title] = 0
                        metadata_by_title[title] = metadata
                    
                    title_counts[title] += 1
                
                # Create display data
                display_data = []
                for title, count in title_counts.items():
                    metadata = metadata_by_title[title]
                    display_data.append({
                        "Title": title,
                        "Chunks": count,
                        "Author": metadata.get("author", "Unknown"),
                        "Type": metadata.get("type", "Unknown"),
                        "Genre": metadata.get("genre", "Unknown"),
                        "Difficulty": metadata.get("difficulty", "Unknown"),
                        "Source Type": metadata.get("source_type", "Unknown"),
                        "Tags": metadata.get("tags", "Unknown")[:50] + "..." if len(str(metadata.get("tags", "Unknown"))) > 50 else metadata.get("tags", "Unknown")
                    })
                
                total_chunks = len(documents)
                st.success(f"📊 Loaded {len(title_counts)} documents with {total_chunks} total chunks from Supabase")
                
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
            
            # Add filtering options
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
    upload_tab1, upload_tab2 = st.tabs(["📄 PDF Upload", "🎥 YouTube URL"])
    
    def add_document_to_csv(title, chunks, author, doc_type, genre, difficulty, source_type, tags):
        """Add a new document entry to the CSV file."""
        try:
            import pandas as pd
            csv_file = "documents_metadata.csv"
            
            # Create new row
            new_row = {
                "title": title,
                "chunks": chunks,
                "author": author,
                "type": doc_type,
                "genre": genre,
                "difficulty": difficulty,
                "source_type": source_type,
                "tags": tags
            }
            
            # Read existing CSV or create new DataFrame
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
            else:
                df = pd.DataFrame(columns=["title", "chunks", "author", "type", "genre", "difficulty", "source_type", "tags"])
            
            # Add new row
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save back to CSV
            df.to_csv(csv_file, index=False)
            return True
        except Exception as e:
            st.error(f"Error updating CSV: {e}")
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
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size} bytes)")
            
            # Metadata form for PDF uploads
            with st.form("pdf_metadata_form"):
                st.subheader("Document Metadata")
                col1, col2 = st.columns(2)
                
                with col1:
                    author = st.text_input("Author", placeholder="Enter author name")
                    doc_type = st.selectbox("Type", ["Book", "Article", "Report", "Paper", "Manual", "Other"])
                    genre = st.selectbox("Genre", ["Fiction", "Non-fiction", "Technical", "Academic", "Business", "Educational", "Other"])
                
                with col2:
                    difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced", "Expert"])
                    tags = st.text_input("Tags", placeholder="comma, separated, tags")
                
                submitted = st.form_submit_button("Process PDF Files", type="primary")
                
                if submitted:
                    for file in uploaded_files:
                        # Extract title from filename (remove .pdf extension)
                        title = file.name.replace('.pdf', '')
                        
                        # TODO: Implement actual PDF processing here
                        # For now, simulate chunk count
                        estimated_chunks = file.size // 1000  # Rough estimate
                        
                        # Add to CSV
                        success = add_document_to_csv(
                            title=title,
                            chunks=estimated_chunks,
                            author=author,
                            doc_type=doc_type,
                            genre=genre,
                            difficulty=difficulty,
                            source_type="PDF",
                            tags=tags
                        )
                        
                        if success:
                            st.success(f"✅ Added {title} to document registry")
                        else:
                            st.error(f"❌ Failed to add {title}")
                    
                    st.info("📝 PDF processing functionality will be implemented to actually process and embed the documents")
    
    with upload_tab2:
        st.subheader("Add YouTube Videos")
        youtube_url = st.text_input(
            "YouTube URL", 
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter a YouTube URL to extract transcript and add to vector store"
        )
        
        if youtube_url:
            # Metadata form for YouTube videos
            with st.form("youtube_metadata_form"):
                st.subheader("Video Metadata")
                col1, col2 = st.columns(2)
                
                with col1:
                    title = st.text_input("Title", placeholder="Video title (auto-detected if empty)")
                    author = st.text_input("Channel/Author", placeholder="Channel name")
                    genre = st.selectbox("Genre", ["Educational", "Tutorial", "Documentary", "Interview", "Lecture", "Other"])
                
                with col2:
                    difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced", "Expert"])
                    tags = st.text_input("Tags", placeholder="comma, separated, tags")
                
                submitted = st.form_submit_button("Process YouTube Video", type="primary")
                
                if submitted:
                    # TODO: Implement actual YouTube processing here
                    # For now, simulate the process
                    if not title:
                        title = f"YouTube Video - {youtube_url.split('=')[-1][:8]}"
                    
                    estimated_chunks = 25  # Rough estimate for video transcript
                    
                    # Add to CSV
                    success = add_document_to_csv(
                        title=title,
                        chunks=estimated_chunks,
                        author=author,
                        doc_type="Video",
                        genre=genre,
                        difficulty=difficulty,
                        source_type="YouTube",
                        tags=tags
                    )
                    
                    if success:
                        st.success(f"✅ Added {title} to document registry")
                    else:
                        st.error(f"❌ Failed to add {title}")
                    
                    st.info("📝 YouTube processing functionality will be implemented to actually extract and embed the transcript")
    
    # CSV Management Section
    st.divider()
    st.subheader("📊 CSV Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 View CSV File"):
            try:
                import pandas as pd
                if os.path.exists("documents_metadata.csv"):
                    df = pd.read_csv("documents_metadata.csv")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("CSV file not found")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    with col2:
        if st.button("🔄 Sync from Supabase"):
            try:
                with st.spinner("Syncing from Supabase..."):
                    result = supabase.table("documents").select("id, metadata").limit(50000).execute()
                    documents = result.data
                    
                    if documents:
                        # Process and save to CSV
                        title_counts = {}
                        metadata_by_title = {}
                        
                        for doc in documents:
                            metadata = doc.get("metadata", {})
                            title = metadata.get("title", "Unknown")
                            
                            if title not in title_counts:
                                title_counts[title] = 0
                                metadata_by_title[title] = metadata
                            
                            title_counts[title] += 1
                        
                        # Create DataFrame
                        data = []
                        for title, count in title_counts.items():
                            metadata = metadata_by_title[title]
                            data.append({
                                "title": title,
                                "chunks": count,
                                "author": metadata.get("author", "Unknown"),
                                "type": metadata.get("type", "Unknown"),
                                "genre": metadata.get("genre", "Unknown"),
                                "difficulty": metadata.get("difficulty", "Unknown"),
                                "source_type": metadata.get("source_type", "Unknown"),
                                "tags": metadata.get("tags", "Unknown")
                            })
                        
                        df = pd.DataFrame(data)
                        df.to_csv("documents_metadata.csv", index=False)
                        st.success(f"✅ Synced {len(df)} documents to CSV")
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

