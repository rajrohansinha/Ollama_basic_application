import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Load environment variables from .env file (Langchain credentials)
load_dotenv()

# Set Langchain credentials and tracing options from environment variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Define a simple chat prompt template for the assistant
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])

def generate_response(question):
    try:
        llm = Ollama(model="llama3.2")
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({'question': question})
    except Exception as e:
        return f"⚠️ Error: {e}"

# --- Streamlit UI ---

st.set_page_config(
    page_title="Ollama Q&A Bot",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS for a fancier look
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
        }
        .stTextInput>div>div>input {
            background-color: #f1f5f9;
            border-radius: 8px;
            border: 1px solid #a5b4fc;
            font-size: 1.1em;
        }
        .stButton>button {
            background-color: #6366f1;
            color: white;
            border-radius: 8px;
            font-size: 1.1em;
            padding: 0.5em 2em;
        }
        .assistant-bubble {
            background: #6366f1;
            color: white;
            padding: 1em;
            border-radius: 12px;
            margin-top: 1em;
            font-size: 1.1em;
            box-shadow: 0 2px 8px rgba(99,102,241,0.08);
        }
        .user-bubble {
            background: #f1f5f9;
            color: #1e293b;
            padding: 1em;
            border-radius: 12px;
            margin-top: 1em;
            font-size: 1.1em;
            border: 1px solid #a5b4fc;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #6366f1;'>🧠 Ollama Q&A Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569;'>Ask anything and get instant answers powered by Ollama LLM!</p>", unsafe_allow_html=True)

# Chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question:", key="input_box", placeholder="Type your question here...")
    submit = st.form_submit_button("Send")

if submit and user_input:
    with st.spinner("Generating response..."):
        response = generate_response(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", response))

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='user-bubble'><b>🧑 You:</b><br>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-bubble'><b>🤖 Assistant:</b><br>{msg}</div>", unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.info("💡 Enter a question above and press 'Send' to start chatting!")