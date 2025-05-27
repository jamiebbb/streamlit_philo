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

# initiating supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize feedback handler
feedback_handler = FeedbackHandler(supabase)

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
    # Sidebar with stats
    with st.sidebar:
        st.header("Feedback Statistics")
        
        stats = feedback_handler.get_feedback_stats()
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Helpful Responses", stats["helpful"])
                st.metric("Partially Helpful", stats["partial"])
            with col2:
                st.metric("Not Helpful", stats["not_helpful"])
                st.metric("Detailed Feedback", stats["detailed"])
        else:
            st.info("No feedback data available yet.")
        
        st.divider()
        st.markdown("### About")
        st.markdown("This is an Agentic RAG chatbot with feedback capabilities.")
        st.markdown("Your feedback helps improve the model's responses!")

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display chat messages from history on app rerun
    for i, message in enumerate(st.session_state.messages):
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
                
                # Add feedback buttons for all except the most recent message
                # This prevents adding feedback buttons to a message the user just received
                if i < len(st.session_state.messages) - 1 and i % 2 == 1:  # Only add to AI messages
                    # Get the corresponding user query (comes before the AI message)
                    user_query = st.session_state.messages[i-1].content if i > 0 else ""
                    feedback_handler.add_feedback_buttons(user_query, message.content)


    # create the bar where we can type messages
    user_question = st.chat_input("How are you?")


    # did the user submit a prompt?
    if user_question:

        # add the message from the user (prompt) to the screen with streamlit
        with st.chat_message("user"):
            st.markdown(user_question)

            st.session_state.messages.append(HumanMessage(user_question))

        # Look for relevant feedback that might contain corrections
        relevant_feedback = feedback_handler.get_relevant_feedback(user_question, embeddings_model=embeddings)
        
        # invoking the agent
        with st.spinner("Thinking..."):
            # Prepare input with feedback if available
            input_data = {"input": user_question, "chat_history": st.session_state.messages}
            
            # If we have relevant feedback, include it in the input
            if relevant_feedback:
                # Add relevant feedback context to the input
                feedback_context = "\n\nIMPORTANT: Users have provided the following corrections to past similar questions:\n"
                for i, feedback in enumerate(relevant_feedback[:3]):  # Limit to top 3 most relevant
                    feedback_context += f"\n{i+1}. Past query: '{feedback['past_query']}'\n"
                    feedback_context += f"   User correction: '{feedback['comment']}'\n"
                
                # Combine the user question with feedback context
                enhanced_question = f"{user_question}\n\n{feedback_context}\n\nPlease take these corrections into account in your response."
                input_data["input"] = enhanced_question
                
                # Add a small note to the UI that feedback is being used
                st.info("📝 Incorporating past feedback into this response", icon="ℹ️")
            
            result = agent_executor.invoke(input_data)

        ai_message = result["output"]

        # adding the response from the llm to the screen (and chat)
        with st.chat_message("assistant"):
            st.markdown(ai_message)
            st.session_state.messages.append(AIMessage(ai_message))
            
            # Add feedback buttons and detailed feedback for the new response
            feedback_handler.add_feedback_buttons(user_question, ai_message)
            
            # Make detailed feedback more prominent when corrections are needed
            st.write("📝 **Notice something incorrect?** Expand below to provide a correction:")
            with st.expander("Provide a correction or detailed feedback"):
                st.write("Your feedback helps improve future answers. If the response contains incorrect information, please explain the correct information below.")
                rating = st.slider("Rate this response (1-5)", 1, 5, 3)
                comment = st.text_area("Correction or comment", placeholder="Explain what's incorrect and provide the right information...")
                
                if st.button("Submit Correction", type="primary"):
                    feedback_handler.store_detailed_feedback(user_question, ai_message, rating, comment)
                    st.success("Thank you for your correction! This will be used to improve future responses.")

    # Add a footer
    st.markdown("---")
    st.markdown("*Your feedback helps us improve! Please rate our responses and provide corrections when needed.*")

# TAB 2: VECTOR STORE VIEWER
with tab2:
    st.header("📚 Vector Store Contents")
    st.write("View all documents currently stored in the vector database.")
    
    # Add refresh button
    if st.button("🔄 Refresh Data", key="refresh_vector_store"):
        st.rerun()
    
    try:
        # Get all documents from Supabase
        with st.spinner("Loading vector store data..."):
            result = supabase.table("documents").select("id, metadata").execute()
            documents = result.data
        
        if not documents:
            st.info("No documents found in the vector store.")
        else:
            st.success(f"Found {len(documents)} total chunks in the vector store")
            
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
            
            # Sort by number of chunks
            display_data.sort(key=lambda x: x["Chunks"], reverse=True)
            
            # Display summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique Documents", len(title_counts))
            with col2:
                st.metric("Total Chunks", len(documents))
            with col3:
                avg_chunks = len(documents) / len(title_counts) if title_counts else 0
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
                
    except Exception as e:
        st.error(f"Error loading vector store data: {e}")

# TAB 3: DOCUMENT UPLOAD
with tab3:
    st.header("📤 Upload Documents")
    st.write("Upload PDF documents or provide YouTube URLs to add to the vector store.")
    
    # Create sub-tabs for different upload types
    upload_tab1, upload_tab2 = st.tabs(["📄 PDF Upload", "🎥 YouTube URL"])
    
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
            
            if st.button("Process PDF Files", type="primary"):
                st.info("PDF processing functionality will be implemented here")
                # TODO: Implement PDF processing
    
    with upload_tab2:
        st.subheader("Add YouTube Videos")
        youtube_url = st.text_input(
            "YouTube URL", 
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter a YouTube URL to extract transcript and add to vector store"
        )
        
        if youtube_url:
            if st.button("Process YouTube Video", type="primary"):
                st.info("YouTube processing functionality will be implemented here")
                # TODO: Implement YouTube processing

