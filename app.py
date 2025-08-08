import os
import uuid
from datetime import datetime
from pathlib import Path
import base64

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# --- LOAD ENV/CONFIG ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INCIDENT_AI_ADMIN_PASS = os.getenv("INCIDENT_AI_ADMIN_PASS", "admin123")
COUNCIL_NAME = os.getenv("COUNCIL_NAME", "Wyndham City Council")
COUNCIL_LOGO = os.getenv("COUNCIL_LOGO", "https://www.wyndham.vic.gov.au/themes/custom/wyndham/logo.png")
COUNCIL_COLOR = os.getenv("COUNCIL_COLOR", "#fe7e36")
LANG = "English"

st.set_page_config(f"{COUNCIL_NAME} – IncidentResponder AI", layout="wide", page_icon="🚨")

# --- SESSION STATE INIT ---
if 'incident_history' not in st.session_state:
    st.session_state.incident_history = []  # list of tuples: (ref, status, sender, msg, file_link, ai_reply, feedback)
if 'role' not in st.session_state:
    st.session_state.role = "Resident"
if 'pdf_index' not in st.session_state:
    st.session_state.pdf_index = None

# --- HELPERS ---
def generate_ref():
    date = datetime.now().strftime("%Y%m%d")
    uid = str(uuid.uuid4())[:6]
    return f"INC-{date}-{uid}"

def get_text_download_link(text, filename):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="{filename}">Download Reply</a>'

# --- BRANDING HEADER ---
def header():
    st.markdown(
        f"""
        <div style='background: linear-gradient(90deg, {COUNCIL_COLOR} 0%, #ff3d3d 100%); border-radius:32px; padding:24px 12px 16px 32px; margin-bottom:12px; display:flex; align-items:center'>
            <img src="{COUNCIL_LOGO}" style="height:60px;border-radius:16px;margin-right:24px;">
            <span style="font-size:40px;font-weight:bold;color:white;vertical-align:middle;margin-left:6px;">
                {COUNCIL_NAME} <span style="font-weight:400;">IncidentResponder AI</span>
            </span>
        </div>
        """, unsafe_allow_html=True
    )

def sidebar():
    with st.sidebar:
        st.image(COUNCIL_LOGO, width=96)
        st.markdown(
            f"""
            <div style="font-size:22px;font-weight:bold;margin-top:8px;margin-bottom:14px;color:{COUNCIL_COLOR}">
                {COUNCIL_NAME}
            </div>
            """, unsafe_allow_html=True
        )
        st.info("IncidentResponder AI helps residents and staff report and resolve incidents or complaints using official council policies.")
        nav = st.radio(
            "Navigate",
            [
                "Incident Assistant",
                "Submit Incident",
                "Incident History",
                "Admin Panel"
            ],
            index=0, key="nav"
        )
        st.markdown("---")
    return nav

def try_asking():
    st.markdown(
        """
        <div style='margin-bottom:18px;'>
            <span style="font-size:18px;color:#fd4c3d;font-weight:600;">💡 Try reporting:</span><br>
        </div>
        """, unsafe_allow_html=True
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.button("Missed bin collection", use_container_width=True, key="q1")
    with col2: st.button("Noise complaint", use_container_width=True, key="q2")
    with col3: st.button("Lost pet", use_container_width=True, key="q3")
    with col4: st.button("Dangerous pothole", use_container_width=True, key="q4")

# --- PDF INDEXING/AI ---
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

def ai_incident_response(incident: str, vectorstore):
    prompt = (
        "You are an Australian council incident and complaint assistant. "
        "Categorize the incoming complaint or incident and reply ONLY using official council policies and FAQs. "
        "Reply with the correct process, any required forms, what to expect next, and if relevant, escalation info."
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever,
        return_source_documents=False
    )
    resp = chain({"query": f"{prompt}\n\nIncident: {incident}"})
    return resp.get("result", "No answer found.")

header()
sidebar_choice = sidebar()

faiss_path = f"index/faiss_index"
pdf_dir = Path("council_docs")
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True, parents=True)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load PDF index at startup
if st.session_state.pdf_index is None:
    if os.path.exists(faiss_path):
        st.session_state.pdf_index = load_faiss_index(faiss_path, embeddings)

# --- Routing ---
if sidebar_choice == "Incident Assistant":
    st.markdown("### 💬 Report an Incident or Complaint")
    try_asking()
    if st.session_state.pdf_index is None:
        st.warning("No official documents indexed yet. Please upload relevant policies/FAQs via the Admin Panel.")
    else:
        st.markdown("#### Incident History (this session)")
        for idx, (ref, status, sender, msg, file_link, ai_reply, feedback) in enumerate(reversed(st.session_state.incident_history)):
            st.markdown(f"**Ref:** {ref} | **Status:** {status}")
            st.markdown(f"<b>{sender}:</b> {msg}", unsafe_allow_html=True)
            if file_link:
                st.markdown(f"<a href='{file_link}' target='_blank'>[File Attached]</a>", unsafe_allow_html=True)
            if ai_reply:
                st.markdown(f"**AI Reply:** {ai_reply}")
                st.markdown(get_text_download_link(ai_reply, f"incident_reply_{ref}.txt"), unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 1, 2])
                if feedback == "":
                    if col1.button("👍", key=f"up_{idx}"):
                        # Update feedback
                        actual_idx = len(st.session_state.incident_history) - 1 - idx
                        old = list(st.session_state.incident_history[actual_idx])
                        old[6] = "up"
                        st.session_state.incident_history[actual_idx] = tuple(old)
                    if col2.button("👎", key=f"down_{idx}"):
                        actual_idx = len(st.session_state.incident_history) - 1 - idx
                        old = list(st.session_state.incident_history[actual_idx])
                        old[6] = "down"
                        st.session_state.incident_history[actual_idx] = tuple(old)
                else:
                    col3.success(f"Feedback: {'👍' if feedback == 'up' else '👎'}")
                if status == "open":
                    if st.button("Escalate to staff", key=f"escalate_{idx}"):
                        actual_idx = len(st.session_state.incident_history) - 1 - idx
                        old = list(st.session_state.incident_history[actual_idx])
                        old[1] = "escalated"
                        st.session_state.incident_history[actual_idx] = tuple(old)
                        st.success("Incident escalated to staff.")
            st.markdown("---")

    inc = st.text_input("Describe your complaint or incident...", key="incident_box")
    file_upload = st.file_uploader("Attach a photo or PDF (optional)", type=["jpg", "jpeg", "png", "pdf"])
    if st.button("Send", key="sendbtn") and inc:
        file_link = ""
        if file_upload is not None:
            file_path = uploads_dir / file_upload.name
            with open(file_path, "wb") as f:
                f.write(file_upload.getbuffer())
            file_link = f"/uploads/{file_upload.name}"
        ref = generate_ref()
        reply = "AI not ready. Please upload council documents." if st.session_state.pdf_index is None else ai_incident_response(inc, st.session_state.pdf_index)
        st.session_state.incident_history.append((ref, "open", "You", inc, file_link, reply, ""))  # feedback is ""
        st.experimental_rerun()

elif sidebar_choice == "Submit Incident":
    st.header("Submit a New Incident/Complaint")
    with st.form("incident_form"):
        inc_msg = st.text_area("Describe your incident or complaint.")
        inc_file = st.file_uploader("Attach a photo or PDF (optional)", type=["jpg", "jpeg", "png", "pdf"])
        inc_email = st.text_input("Your email for follow-up (optional)")
        sent = st.form_submit_button("Submit")
        if sent and inc_msg:
            file_link = ""
            if inc_file is not None:
                file_path = uploads_dir / inc_file.name
                with open(file_path, "wb") as f:
                    f.write(inc_file.getbuffer())
                file_link = f"/uploads/{inc_file.name}"
            ref = generate_ref()
            # AI reply can be generated here or by staff
            st.session_state.incident_history.append((ref, "open", "You", inc_msg, file_link, "", ""))
            st.success(f"Your incident (Ref: {ref}) was submitted. If unresolved, staff will follow up.")

elif sidebar_choice == "Incident History":
    st.header("Incident History & Session Log")
    st.write(f"Incidents reported: {len(st.session_state.incident_history)}")
    df = pd.DataFrame(st.session_state.incident_history,
        columns=["Reference", "Status", "Sender", "Message", "File", "AI Reply", "Feedback"])
    st.dataframe(df, use_container_width=True)
    if len(df) > 0:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="incident_log.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

elif sidebar_choice == "Admin Panel":
    st.header("Admin Panel")
    if st.session_state.role != "Admin":
        pwd = st.text_input("Enter admin password", type="password")
        if st.button("Login as admin"):
            if pwd == INCIDENT_AI_ADMIN_PASS:
                st.session_state.role = "Admin"
                st.success("Welcome, admin.")
                st.experimental_rerun()
            else:
                st.error("Incorrect password.")
    else:
        st.write(f"Upload Council PDFs (policies, FAQs, forms)")
        uploaded_pdfs = st.file_uploader("Upload multiple PDFs", accept_multiple_files=True, type="pdf")
        if uploaded_pdfs:
            pdf_dir.mkdir(parents=True, exist_ok=True)
            for pdf in uploaded_pdfs:
                with open(pdf_dir / pdf.name, "wb") as f:
                    f.write(pdf.getbuffer())
            st.session_state.pdf_index = build_pdf_index(pdf_dir, faiss_path)
            st.success("PDFs indexed for incident response! Return to Assistant to try it out.")
        if st.button("Reset Session"):
            st.session_state.incident_history = []
            st.session_state.pdf_index = None
            st.success("Session reset.")

st.markdown(f"""
<br>
<div style='font-size:13px;text-align:center;color:#aaa'>Made with 🚨 IncidentResponder AI – for {COUNCIL_NAME}, powered by AI</div>
""", unsafe_allow_html=True)
