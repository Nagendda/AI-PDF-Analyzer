# app.py

import streamlit as st
from dotenv import load_dotenv
import os
from pdf_processor import extract_text_from_pdf, get_text_chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()

def get_vector_store(text_chunks):
    """Creates and returns a vector store from text chunks."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

def get_conversational_chain():
    """Creates and returns a question-answering chain."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context". Don't provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Handles user input, performs similarity search, and gets an answer."""
    if "vector_store" not in st.session_state or not st.session_state.vector_store:
        st.warning("Please upload and process a PDF first.")
        return

    vector_store = st.session_state.vector_store
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("### Answer")
    st.write(response["output_text"])

# --- Streamlit App Interface ---
st.set_page_config(page_title="DocuChat", layout="wide")

st.header("📄 DocuChat: Chat with Your PDF")

# Initialize session state for vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.title("Your Document")
    pdf_doc = st.file_uploader("Upload your PDF file and click 'Process'", type="pdf")
    if st.button("Process"):
        if pdf_doc:
            with st.spinner("Processing PDF..."):
                # 1. Extract text
                raw_text = extract_text_from_pdf(pdf_doc)
                if raw_text.startswith("Error"):
                    st.error(raw_text)
                else:
                    # 2. Split into chunks
                    text_chunks = get_text_chunks(raw_text)

                    # 3. Create vector store and save to session
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    if st.session_state.vector_store:
                        st.success("Processing complete! You can now ask questions.")
        else:
            st.warning("Please upload a PDF file.")

# Main chat interface
st.subheader("Ask a Question about Your Document")
question = st.text_input("Enter your question:")
if question:
    user_input(question)