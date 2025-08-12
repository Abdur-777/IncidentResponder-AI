# =========================
# IncidentResponder AI — Wyndham
# Staff assistant over council policy knowledge
# Region: AU (australia-southeast1)
# Features merged:
#  - Save uploaded file to KB (GCS) + auto reindex
#  - Admin panel: Rebuild index + Logs
#  - RAG over FAISS index stored in GCS
# =========================

import os, io, re, json, time, shutil, datetime, tempfile, traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd

# LangChain / OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA

# GCS
from google.cloud import storage

# -------------- App Config --------------
st.set_page_config(page_title="Wyndham — IncidentResponder AI", page_icon="🏛️", layout="wide")

APP_TITLE = "Wyndham — IncidentResponder AI"
DATA_REGION = os.getenv("DATA_REGION", "australia-southeast1")

# Council + Storage
COUNCIL_KEY = os.getenv("COUNCIL_KEY", "wyndham")
GCS_BUCKET = os.getenv("GCS_BUCKET", "civreply-data")
GCS_DOCS_PREFIX = f"policies/{COUNCIL_KEY}"
GCS_INDEX_PREFIX = f"{GCS_DOCS_PREFIX}/_hash_index"
GCS_LOGS_PREFIX = f"{GCS_DOCS_PREFIX}/_logs"
GCS_MANIFEST_BLOB = f"{GCS_INDEX_PREFIX}/manifest.json"

# Admin
ADMIN_PIN = os.getenv("ADMIN_PIN", "4242")

# Model
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Auth: if running on Render/Cloud, write SA JSON to tmp path
if "GCP_SA_JSON" in os.environ and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    with open("/tmp/gcs-key.json", "w") as f:
        f.write(os.environ["GCP_SA_JSON"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcs-key.json"

# -------------- Utilities --------------
def gcs_client() -> storage.Client:
    return storage.Client()

def _ensure_tmp_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _download_gcs_dir(bucket: storage.Bucket, prefix: str, local_dir: Path):
    if local_dir.exists():
        shutil.rmtree(local_dir)
    _ensure_tmp_dir(local_dir)
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix):].lstrip("/")
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))

def _upload_dir_to_gcs(bucket: storage.Bucket, local_dir: Path, gcs_prefix: str):
    for root, _, files in os.walk(local_dir):
        for f in files:
            p = Path(root) / f
            rel = str(p.relative_to(local_dir))
            bucket.blob(f"{gcs_prefix}/{rel}").upload_from_filename(str(p))

def _upload_bytes_to_gcs(bucket: storage.Bucket, dest_path: str, payload: bytes, content_type="application/pdf"):
    blob = bucket.blob(dest_path)
    blob.cache_control = "no-cache"
    blob.upload_from_string(payload, content_type=content_type)

def _write_manifest(bucket: storage.Bucket, info: Dict[str, Any]):
    bucket.blob(GCS_MANIFEST_BLOB).upload_from_string(json.dumps(info, indent=2), content_type="application/json")

def _read_manifest(bucket: storage.Bucket) -> Optional[Dict[str, Any]]:
    blob = bucket.blob(GCS_MANIFEST_BLOB)
    if not blob.exists():
        return None
    try:
        return json.loads(blob.download_as_text())
    except Exception:
        return None

def log_event(event_type: str, payload: Dict[str, Any]):
    try:
        client = gcs_client(); bucket = client.bucket(GCS_BUCKET)
        ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S-%fZ")
        blob = bucket.blob(f"{GCS_LOGS_PREFIX}/{ts}-{event_type}.json")
        blob.upload_from_string(json.dumps({"ts": ts, "type": event_type, **payload}), content_type="application/json")
    except Exception:
        # Avoid UI crashes on logging failure
        pass

# -------------- Index Helpers --------------
def rebuild_index_from_gcs(include_patterns=(".pdf", ".PDF")) -> Dict[str, Any]:
    """
    Pull PDFs from GCS -> /tmp, build FAISS, push index back to GCS, write manifest.
    """
    t0 = time.time()
    client = gcs_client()
    bucket = client.bucket(GCS_BUCKET)

    # 1) Download PDFs
    tmp_docs = Path("/tmp/kb_docs")
    _download_gcs_dir(bucket, f"{GCS_DOCS_PREFIX}/", tmp_docs)

    # 2) Load + split
    loader = PyPDFDirectoryLoader(str(tmp_docs))
    docs = loader.load()
    # Filter by extension (safety)
    docs = [d for d in docs if any(str(d.metadata.get("source", "")).endswith(ext) for ext in include_patterns)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)
    split_docs = splitter.split_documents(docs)

    # 3) Build FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    db = FAISS.from_documents(split_docs, embeddings)

    # 4) Save locally and push to GCS
    tmp_index = Path("/tmp/faiss_index")
    if tmp_index.exists():
        shutil.rmtree(tmp_index)
    db.save_local(str(tmp_index))

    # Clear previous index path in GCS, then upload
    for b in list(bucket.list_blobs(prefix=f"{GCS_INDEX_PREFIX}/")):
        b.delete()
    _upload_dir_to_gcs(bucket, tmp_index, GCS_INDEX_PREFIX)

    dt = datetime.datetime.utcnow().isoformat() + "Z"
    manifest = {
        "council": COUNCIL_KEY,
        "gcs_bucket": GCS_BUCKET,
        "gcs_docs_prefix": GCS_DOCS_PREFIX,
        "gcs_index_prefix": GCS_INDEX_PREFIX,
        "docs_count": len(split_docs),
        "updated_at_utc": dt,
        "build_seconds": round(time.time() - t0, 2),
    }
    _write_manifest(bucket, manifest)
    log_event("reindex", manifest)
    return manifest

def load_faiss_from_gcs() -> Optional[FAISS]:
    """
    Download FAISS index folder from GCS to /tmp/current_index and return a loaded FAISS store.
    """
    client = gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    tmp_index = Path("/tmp/current_index")
    _download_gcs_dir(bucket, f"{GCS_INDEX_PREFIX}/", tmp_index)

    # FAISS expects two files: index.faiss and index.pkl
    faiss_bin = tmp_index / "index.faiss"
    faiss_pkl = tmp_index / "index.pkl"
    if not (faiss_bin.exists() and faiss_pkl.exists()):
        return None

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    return FAISS.load_local(str(tmp_index), embeddings, allow_dangerous_deserialization=True)

def upload_uploaded_file_to_kb(uploaded_file, subdir="uploads") -> str:
    """
    Save an uploaded file into gs://<bucket>/policies/<council>/<subdir>/... and return its GCS path.
    """
    client = gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    safe_name = uploaded_file.name.replace("/", "_")
    dest = f"{GCS_DOCS_PREFIX}/{subdir}/{ts}-{safe_name}"
    _upload_bytes_to_gcs(bucket, dest, uploaded_file.getvalue(), content_type=uploaded_file.type or "application/pdf")
    log_event("upload_to_kb", {"dest": dest, "size_bytes": len(uploaded_file.getvalue())})
    return dest

# -------------- RAG --------------
def answer_with_retriever(retriever, question: str) -> Dict[str, Any]:
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, streaming=False)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    result = chain({"query": question})
    return result

def render_sources(docs: List[Document]):
    if not docs:
        return
    with st.expander("Sources", expanded=False):
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", None)
            pg = f" • p.{page+1}" if isinstance(page, int) else ""
            st.markdown(f"{i}. `{os.path.basename(src)}`{pg}")

# -------------- UI: Header --------------
def header():
    client = gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    manifest = _read_manifest(bucket)
    ts = manifest.get("updated_at_utc") if manifest else "—"
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.title(APP_TITLE)
        st.caption(f"Staff assistant with council policy knowledge. • Data region: {DATA_REGION}")
    with col2:
        st.markdown(
            f"<div style='text-align:right;color:#6b7280;'>Last indexed<br><b>{ts}</b></div>",
            unsafe_allow_html=True,
        )

# -------------- UI: Footer Info --------------
def footer_info():
    st.markdown(
        f"<hr style='opacity:0.2'/>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Council: **{COUNCIL_KEY}**  •  Bucket: **{GCS_BUCKET}**  •  Docs: **{GCS_DOCS_PREFIX}**  •  "
        f"Index: **{GCS_INDEX_PREFIX}**  •  Data region: **{DATA_REGION}**"
    )
    st.caption("© 2025 IncidentResponder AI")

# -------------- UI: Admin Sidebar --------------
def admin_sidebar():
    with st.sidebar.expander("🔐 Admin", expanded=False):
        pin = st.text_input("Enter admin PIN", type="password")
        if st.button("Unlock"):
            st.session_state["admin"] = (pin == ADMIN_PIN)

        if st.session_state.get("admin"):
            st.success("Admin unlocked")

            if st.button("🔄 Rebuild index now"):
                with st.spinner("Rebuilding…"):
                    info = rebuild_index_from_gcs()
                st.toast("Index rebuilt")
                st.json(info)

            if st.button("📜 Refresh logs"):
                pass

            # Display latest 200 logs
            try:
                client = gcs_client(); bucket = client.bucket(GCS_BUCKET)
                blobs = sorted(
                    [b for b in bucket.list_blobs(prefix=f"{GCS_LOGS_PREFIX}/")],
                    key=lambda b: b.name,
                    reverse=True
                )[:200]
                rows = []
                for b in blobs:
                    try:
                        rows.append(json.loads(b.download_as_text()))
                    except Exception:
                        continue
                df = pd.DataFrame(rows)
                if not df.empty:
                    st.dataframe(df, use_container_width=True, height=300)
                else:
                    st.caption("No logs yet.")
            except Exception as e:
                st.warning(f"Could not load logs: {e}")
        else:
            st.info("Enter PIN to access admin tools.")

# -------------- UI: Main Modes --------------
def kb_mode():
    """
    Knowledge Base mode: use prebuilt FAISS index from GCS.
    """
    st.radio("Choose a data source", ["Knowledge Base", "Upload a File"], index=0, key="ds_kb", horizontal=True)
    st.markdown("**Ask a question:**")
    q = st.text_input("", placeholder="Type your question…", label_visibility="collapsed")

    if "kb" not in st.session_state:
        st.session_state["kb"] = {"retriever": None, "loaded": False, "err": None}

    # Lazy-load FAISS from GCS
    if not st.session_state["kb"]["loaded"]:
        with st.spinner("Loading knowledge base index…"):
            try:
                db = load_faiss_from_gcs()
                if not db:
                    st.session_state["kb"]["err"] = "No index found in GCS. Ask an admin to rebuild."
                else:
                    st.session_state["kb"]["retriever"] = db.as_retriever(search_kwargs={"k": TOP_K})
                    st.session_state["kb"]["loaded"] = True
            except Exception as e:
                st.session_state["kb"]["err"] = str(e)

    if st.session_state["kb"]["err"]:
        st.error(st.session_state["kb"]["err"])
        return

    if q:
        try:
            with st.spinner("Thinking…"):
                result = answer_with_retriever(st.session_state["kb"]["retriever"], q)
            st.markdown(result["result"])
            render_sources(result.get("source_documents", []))
            log_event("qa_kb", {"q": q})
        except Exception as e:
            st.error(f"Error answering: {e}")
            st.caption("Check logs / admin panel for details.")
            log_event("qa_kb_error", {"q": q, "error": str(e)})

def upload_mode():
    """
    Upload a File mode:
      - Optionally persist to KB + rebuild index
      - Else ephemeral, indexed only in-session
    """
    st.radio("Choose a data source", ["Knowledge Base", "Upload a File"], index=1, key="ds_up", horizontal=True)
    up = st.file_uploader("Upload a PDF", type=["pdf"])
    save_to_kb = st.checkbox("Save uploaded file to Knowledge Base (and reindex)", value=True)

    if "ephemeral_db" not in st.session_state:
        st.session_state["ephemeral_db"] = None
    if "ephemeral_retriever" not in st.session_state:
        st.session_state["ephemeral_retriever"] = None

    if up and st.button("Process file"):
        if save_to_kb:
            try:
                dest = upload_uploaded_file_to_kb(up)
                with st.spinner("Rebuilding index…"):
                    info = rebuild_index_from_gcs()
                st.success(f"Saved to GCS and reindexed.\nPath: gs://{GCS_BUCKET}/{dest}")
                st.caption(f"Index: gs://{GCS_BUCKET}/{GCS_INDEX_PREFIX}")
                st.json(info)
            except Exception as e:
                st.error(f"Could not save/reindex: {e}")
        else:
            # Ephemeral local index (session-only)
            try:
                tmp_dir = Path("/tmp/session_upload")
                if tmp_dir.exists(): shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                fpath = tmp_dir / up.name.replace("/", "_")
                with open(fpath, "wb") as f:
                    f.write(up.getvalue())

                loader = PyPDFDirectoryLoader(str(tmp_dir))
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)
                split_docs = splitter.split_documents(docs)
                embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
                db = FAISS.from_documents(split_docs, embeddings)
                st.session_state["ephemeral_db"] = db
                st.session_state["ephemeral_retriever"] = db.as_retriever(search_kwargs={"k": TOP_K})
                st.success("Loaded file for this session. Ask away below.")
            except Exception as e:
                st.error(f"Could not process file: {e}")

    if (st.session_state.get("ephemeral_retriever") is not None) or save_to_kb:
        st.markdown("**Ask a question about the uploaded content:**")
        q = st.text_input("Your question", placeholder="e.g., What obligations apply to residents?")
        if q:
            try:
                retriever = (
                    st.session_state["ephemeral_retriever"]
                    if st.session_state.get("ephemeral_retriever") is not None
                    else None
                )
                if retriever is None:
                    # If saved to KB, we can answer via the (now) rebuilt KB
                    db = load_faiss_from_gcs()
                    if not db:
                        st.error("KB index not available yet; try again in a moment or contact admin.")
                        return
                    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

                with st.spinner("Thinking…"):
                    result = answer_with_retriever(retriever, q)
                st.markdown(result["result"])
                render_sources(result.get("source_documents", []))
                log_event("qa_upload", {"q": q, "persisted": save_to_kb})
            except Exception as e:
                st.error(f"Error answering: {e}")
                log_event("qa_upload_error", {"q": q, "error": str(e)})

# -------------- Main --------------
def main():
    header()
    admin_sidebar()

    # Mode switcher
    tabs = st.tabs(["Knowledge Base", "Upload a File"])
    with tabs[0]:
        kb_mode()
    with tabs[1]:
        upload_mode()

    footer_info()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Unexpected error in app.")
        st.code(traceback.format_exc())
