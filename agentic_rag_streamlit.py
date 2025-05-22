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
llm = ChatOpenAI(model="gpt-4o",temperature=0)

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

# Create a unique session ID if not exists
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
    
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


    # invoking the agent
    with st.spinner("Thinking..."):
        result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})

    ai_message = result["output"]

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(ai_message)
        st.session_state.messages.append(AIMessage(ai_message))
        
        # Add feedback buttons and detailed feedback for the new response
        feedback_handler.add_feedback_buttons(user_question, ai_message)
        feedback_handler.add_detailed_feedback(user_question, ai_message)

# Add a footer
st.markdown("---")
st.markdown("*Your feedback helps us improve! Please rate our responses.*")

