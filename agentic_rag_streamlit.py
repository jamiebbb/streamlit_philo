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
                
                # Generate metadata button for each file
                if st.button(f"🤖 Generate Metadata for {file.name}", key=f"generate_{file_key}"):
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
                            Analyze this document and generate metadata. Document title: "{file.name.replace('.pdf', '')}"
                            
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
                                
                            except Exception as parse_error:
                                st.error(f"Error parsing metadata: {parse_error}")
                                # Fallback metadata
                                st.session_state[f"metadata_{file_key}"] = {
                                    "title": file.name.replace('.pdf', ''),
                                    "author": "Unknown",
                                    "summary": "Document summary not available",
                                    "type": "Document",
                                    "genre": "Other",
                                    "topic": "General",
                                    "tags": "document, pdf",
                                    "difficulty": "Intermediate"
                                }
                                st.warning("Using fallback metadata. Please review and edit as needed.")
                                
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
                                title = st.text_input("Title", value=metadata.get("title", ""), key=f"title_{file_key}")
                                author = st.text_input("Author", value=metadata.get("author", ""), key=f"author_{file_key}")
                                doc_type = st.selectbox("Type", 
                                    ["Book", "Article", "Report", "Paper", "Manual", "Guide", "Other"],
                                    index=["Book", "Article", "Report", "Paper", "Manual", "Guide", "Other"].index(metadata.get("type", "Other")) if metadata.get("type") in ["Book", "Article", "Report", "Paper", "Manual", "Guide", "Other"] else 6,
                                    key=f"type_{file_key}")
                                genre = st.selectbox("Genre", 
                                    ["Fiction", "Non-fiction", "Technical", "Academic", "Business", "Educational", "Legal", "Medical", "Other"],
                                    index=["Fiction", "Non-fiction", "Technical", "Academic", "Business", "Educational", "Legal", "Medical", "Other"].index(metadata.get("genre", "Other")) if metadata.get("genre") in ["Fiction", "Non-fiction", "Technical", "Academic", "Business", "Educational", "Legal", "Medical", "Other"] else 8,
                                    key=f"genre_{file_key}")
                            
                            with col2:
                                topic = st.text_input("Topic", value=metadata.get("topic", ""), key=f"topic_{file_key}")
                                difficulty = st.selectbox("Difficulty", 
                                    ["Beginner", "Intermediate", "Advanced", "Expert"],
                                    index=["Beginner", "Intermediate", "Advanced", "Expert"].index(metadata.get("difficulty", "Intermediate")) if metadata.get("difficulty") in ["Beginner", "Intermediate", "Advanced", "Expert"] else 1,
                                    key=f"difficulty_{file_key}")
                                tags = st.text_input("Tags", value=metadata.get("tags", ""), key=f"tags_{file_key}")
                            
                            summary = st.text_area("Summary", value=metadata.get("summary", ""), height=100, key=f"summary_{file_key}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.form_submit_button("✅ Process & Upload Document", type="primary"):
                                    # Process the document
                                    success = process_pdf_document(
                                        file, title, author, summary, doc_type, genre, topic, tags, difficulty
                                    )
                                    if success:
                                        st.success(f"🎉 Successfully processed {title}!")
                                        # Clear the metadata from session state
                                        del st.session_state[f"metadata_{file_key}"]
                                        st.rerun()
                            
                            with col2:
                                if st.form_submit_button("🗑️ Cancel", type="secondary"):
                                    # Clear the metadata from session state
                                    del st.session_state[f"metadata_{file_key}"]
                                    st.rerun()
    
    def process_pdf_document(file, title, author, summary, doc_type, genre, topic, tags, difficulty):
        """Process PDF document: extract text, chunk, embed, and save to both CSV and Supabase."""
        try:
            # Reset file pointer
            file.seek(0)
            
            # Extract text from PDF
            from pypdf import PdfReader
            import io
            from langchain.text_splitter import RecursiveCharacterTextSplitter
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
            
            # Create document metadata
            metadata = {
                "title": title,
                "author": author,
                "summary": summary,
                "type": doc_type,
                "genre": genre,
                "topic": topic,
                "tags": tags,
                "difficulty": difficulty,
                "source_type": "PDF",
                "source": f"PDF: {file.name}"
            }
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
            )
            
            # Create document object
            doc = Document(page_content=full_text, metadata=metadata)
            
            # Split into chunks
            chunks = text_splitter.split_documents([doc])
            
            # Add to vector store (Supabase)
            with st.spinner(f"Embedding {len(chunks)} chunks..."):
                vector_store.add_documents(chunks)
            
            # Add to CSV
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
            
            if success:
                st.success(f"✅ Added {len(chunks)} chunks to vector store and updated CSV")
                return True
            else:
                st.error("Failed to update CSV")
                return False
                
        except Exception as e:
            st.error(f"Error processing document: {e}")
            return False

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
                
                # Generate metadata button
                if st.button(f"🤖 Generate Metadata for YouTube Video", key="generate_youtube"):
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
                                    video_metadata.get("title", "Unknown"), 
                                    cleaned_transcript, 
                                    video_metadata
                                )
                                
                                # Store in session state
                                st.session_state["youtube_metadata"] = enhanced_metadata
                                st.session_state["youtube_transcript"] = cleaned_transcript
                                st.session_state["youtube_video_id"] = video_id
                                
                                st.success("✅ Metadata and transcript generated successfully!")
                                
                        except Exception as e:
                            st.error(f"Error processing YouTube video: {e}")
                
                # Show metadata form if generated
                if "youtube_metadata" in st.session_state:
                    metadata = st.session_state["youtube_metadata"]
                    
                    with st.expander("📝 Review & Edit YouTube Video Metadata", expanded=True):
                        with st.form("youtube_metadata_form"):
                            st.subheader("Generated Metadata - Please Review & Edit")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                title = st.text_input("Title", value=metadata.get("title", ""), key="youtube_title")
                                author = st.text_input("Author/Speaker", value=metadata.get("author", ""), key="youtube_author")
                                youtube_channel = st.text_input("YouTube Channel", value=metadata.get("youtube_channel", ""), key="youtube_channel", disabled=True)
                                genre = st.selectbox("Genre", 
                                    ["Educational", "Tutorial", "Documentary", "Interview", "Lecture", "Entertainment", "News", "Other"],
                                    index=["Educational", "Tutorial", "Documentary", "Interview", "Lecture", "Entertainment", "News", "Other"].index(metadata.get("genre", "Educational")) if metadata.get("genre") in ["Educational", "Tutorial", "Documentary", "Interview", "Lecture", "Entertainment", "News", "Other"] else 0,
                                    key="youtube_genre")
                            
                            with col2:
                                topic = st.text_input("Topic", value=metadata.get("topic", ""), key="youtube_topic")
                                difficulty = st.selectbox("Difficulty", 
                                    ["Beginner", "Intermediate", "Advanced", "Expert"],
                                    index=["Beginner", "Intermediate", "Advanced", "Expert"].index(metadata.get("difficulty", "Intermediate")) if metadata.get("difficulty") in ["Beginner", "Intermediate", "Advanced", "Expert"] else 1,
                                    key="youtube_difficulty")
                                tags = st.text_input("Tags", value=metadata.get("tags", ""), key="youtube_tags")
                            
                            summary = st.text_area("Summary", value=metadata.get("summary", ""), height=100, key="youtube_summary")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.form_submit_button("✅ Process & Upload Video", type="primary"):
                                    # Process the YouTube video
                                    success = process_youtube_video(
                                        st.session_state["youtube_video_id"],
                                        st.session_state["youtube_transcript"],
                                        title, author, summary, genre, topic, tags, difficulty,
                                        st.session_state["youtube_metadata"]
                                    )
                                    if success:
                                        st.success(f"🎉 Successfully processed {title}!")
                                        # Clear the metadata from session state
                                        for key in ["youtube_metadata", "youtube_transcript", "youtube_video_id"]:
                                            if key in st.session_state:
                                                del st.session_state[key]
                                        st.rerun()
                            
                            with col2:
                                if st.form_submit_button("🗑️ Cancel", type="secondary"):
                                    # Clear the metadata from session state
                                    for key in ["youtube_metadata", "youtube_transcript", "youtube_video_id"]:
                                        if key in st.session_state:
                                            del st.session_state[key]
                                    st.rerun()

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

    def process_youtube_video(video_id, transcript, title, author, summary, genre, topic, tags, difficulty, original_metadata):
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
            
            # Split text into chunks (400/200 for YouTube videos)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=200,
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
            
            # Add to CSV
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
            
            if success:
                st.success(f"✅ Added {len(chunks)} chunks to vector store and updated CSV")
                return True
            else:
                st.error("Failed to update CSV")
                return False
                
        except Exception as e:
            st.error(f"Error processing YouTube video: {e}")
            return False

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

