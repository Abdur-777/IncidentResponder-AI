import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# =========================
#  LOAD ENVIRONMENT VARS
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Please set it in your environment variables.")
    st.stop()

# =========================
#  STREAMLIT PAGE CONFIG
# =========================
st.set_page_config(
    page_title="IncidentResponder AI",
    page_icon="🚨",
    layout="wide"
)

# =========================
#  HEADER
# =========================
st.title("🚨 IncidentResponder AI")
st.markdown("### AI-Powered Complaint & Incident Report Assistant")
st.write("Upload incident/complaint files (PDF) and get instant AI-generated summaries and responses.")

# =========================
#  FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

# =========================
#  USER QUESTION INPUT
# =========================
user_query = st.text_input("💬 Enter a question about the uploaded document:")

# =========================
#  PROCESS FILE
# =========================
if uploaded_file:
    # Read PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text.strip():
        st.error("⚠ Could not extract text from this PDF. Please try another file.")
        st.stop()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings & FAISS vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # If a user asks a question
    if user_query:
        with st.spinner("🤖 Generating AI response..."):
            llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0,
                openai_api_key=OPENAI_API_KEY
            )

            retriever = vectorstore.as_retriever()
            docs = retriever.get_relevant_documents(user_query)
            context = "\n".join([doc.page_content for doc in docs])

            prompt = f"""
            You are an AI assistant for handling incident and complaint reports.

            Context from document:
            {context}

            Question:
            {user_query}

            Provide a clear, professional, and accurate response.
            """
            answer = llm.predict(prompt)

        st.subheader("📌 AI Response")
        st.write(answer)

        # Show relevant document extracts
        with st.expander("📄 Relevant Extracts from Document"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Extract {i}:**")
                st.write(doc.page_content)
else:
    st.info("⬆ Please upload a PDF to begin.")
