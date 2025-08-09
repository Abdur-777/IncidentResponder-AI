import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# === LOAD ENV ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Please set it in your hosting platform's secrets and restart the app.")
    st.stop()

# === CONFIG ===
COUNCIL_NAME = "Wyndham City Council"
COUNCIL_EMAIL = "civreplywyndham@gmail.com"
GCS_BUCKET = os.getenv("GCS_BUCKET")  # Optional, for cloud storage

st.set_page_config(page_title="IncidentResponder AI", page_icon="🚨", layout="wide")

# === GCS HELPERS ===
def upload_to_gcs(local_file, bucket_name, remote_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_file)

def download_from_gcs(bucket_name, remote_path, local_file):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(remote_path)
    if blob.exists():
        blob.download_to_filename(local_file)
        return True
    return False

# === PDF INDEXING ===
def build_pdf_index(pdf_dir: Path, faiss_index_path: str):
    loader = PyPDFDirectoryLoader(str(pdf_dir))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)
    split_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(faiss_index_path)
    return vectorstore

def load_faiss_index(faiss_index_path: str, embeddings):
    if os.path.exists(faiss_index_path):
        return FAISS.load_local(faiss_index_path, embeddings)
    return None

# === AI QA ===
def ai_qa(question: str, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever,
        return_source_documents=False
    )
    resp = chain({"query": question})
    return resp.get("result", "No answer found.")

# === HEADER ===
st.markdown(
    f"""
    <div style='background:#ff4d4d; border-radius:12px; padding:16px; color:white;'>
        <h1>🚨 {COUNCIL_NAME} - IncidentResponder AI</h1>
        <p>Automatically respond to complaints & incidents, powered by council documents.</p>
    </div>
    """, unsafe_allow_html=True
)

# === SIDEBAR ===
menu = st.sidebar.radio("Navigation", ["Chat", "Report Incident", "Admin Panel", "About"])
st.sidebar.markdown("---")
st.sidebar.info(f"Connected Council: {COUNCIL_NAME}")

# === SESSION STATE ===
if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Paths ===
faiss_path = f"index/{COUNCIL_NAME.lower().replace(' ', '_')}_index"
pdf_dir = Path(f"council_docs/{COUNCIL_NAME.lower().replace(' ', '_')}")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# === Load FAISS Index if Exists ===
if st.session_state.pdf_index is None:
    if os.path.exists(faiss_path):
        st.session_state.pdf_index = load_faiss_index(faiss_path, embeddings)
    elif GCS_BUCKET:
        if download_from_gcs(GCS_BUCKET, f"faiss_indexes/{COUNCIL_NAME.lower()}", faiss_path):
            st.session_state.pdf_index = load_faiss_index(faiss_path, embeddings)

# === CHAT PAGE ===
if menu == "Chat":
    st.subheader("💬 Ask the AI")
    if st.session_state.pdf_index is None:
        st.warning("No council documents indexed yet. Admin must upload files.")
    else:
        for sender, msg in st.session_state.chat_history:
            st.markdown(f"**{sender}:** {msg}")
        user_q = st.text_input("Type your question here...")
        if st.button("Send") and user_q:
            st.session_state.chat_history.append(("You", user_q))
            answer = ai_qa(user_q, st.session_state.pdf_index)
            st.session_state.chat_history.append(("AI", answer))
            st.experimental_rerun()

# === REPORT INCIDENT ===
elif menu == "Report Incident":
    st.subheader("📢 Report an Incident or Complaint")
    with st.form("incident_form"):
        desc = st.text_area("Describe the incident")
        file = st.file_uploader("Upload evidence (photo/document)", type=["jpg", "jpeg", "png", "pdf"])
        submitted = st.form_submit_button("Submit")
        if submitted and desc:
            st.success("Incident submitted successfully. Council will respond shortly.")

# === ADMIN PANEL ===
elif menu == "Admin Panel":
    pwd = st.text_input("Enter admin password", type="password")
    if pwd == os.getenv("ADMIN_PASS", "admin123"):
        st.success("Admin Access Granted")
        uploaded_pdfs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")
        if uploaded_pdfs:
            pdf_dir.mkdir(parents=True, exist_ok=True)
            for pdf in uploaded_pdfs:
                with open(pdf_dir / pdf.name, "wb") as f:
                    f.write(pdf.getbuffer())
            st.session_state.pdf_index = build_pdf_index(pdf_dir, faiss_path)
            if GCS_BUCKET:
                upload_to_gcs(faiss_path, GCS_BUCKET, f"faiss_indexes/{COUNCIL_NAME.lower()}")
            st.success("PDFs indexed successfully!")
    else:
        if pwd:
            st.error("Incorrect password.")

# === ABOUT ===
elif menu == "About":
    st.info("IncidentResponder AI helps councils respond to incidents and complaints instantly using AI.")
