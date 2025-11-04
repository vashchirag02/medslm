import streamlit as st
from medslm_pipeline import retrieve_and_generate_answer

# Page setup
st.set_page_config(
    page_title="🩺 MedSLM Chatbot",
    page_icon="💬",
    layout="centered"
)

# Sidebar
st.sidebar.title("⚙️ About MedSLM")
st.sidebar.info("""
This medical chatbot is powered by a **Specialized Language Model (SLM)**  
trained on **20,000 medical research abstracts** using:
- 🔍 FAISS for document retrieval  
- 🧠 Sentence Transformers for embeddings  
- 🧩 Groq LLM (Llama 3.3 70B) for reasoning  
""")

st.title("🩺 MedSLM - AI Medical Research Assistant")
st.caption("Ask me anything related to medical research, treatments, or diseases.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_query = st.chat_input("Type your medical question here...")

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(message)
    else:
        with st.chat_message("assistant"):
            st.markdown(message)

# Process new user query
if user_query:
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query and retrieving medical insights..."):
            try:
                answer = retrieve_and_generate_answer(user_query)
                st.markdown(answer)
                st.session_state.chat_history.append(("assistant", answer))
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
