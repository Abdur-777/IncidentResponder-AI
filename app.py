# =========================
# IncidentResponder AI — Wyndham
# Staff assistant over council policy knowledge
# Region: AU (australia-southeast1)
# Features:
#  - Save uploaded file to KB (GCS) + auto reindex
#  - Admin panel: Rebuild index (PDF-only or multi-format via script) + Logs
#  - RAG over FAISS index stored in GCS
# =========================

import os, io, re, json, time, shutil, datetime, tempfile, traceback, subprocess, shlex
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

# Council + Storage (defaults aligned to gs://civreply-data/policies/{slug}/_hash_index)
COUNCIL_KEY = os.getenv("COUNCIL_KEY", "wyndham")
GCS_BUCKET  = os.getenv("GCS_BUCKET", "civreply-data")

# Prefixes (docs + index)
DOC_PREFIX   = os.getenv("DOC_PREFIX",  "policies/{slug}")                  # where documents live
INDEX_PREFIX = os.getenv("INDEX_PREFIX","policies/{slug}/_hash_index")      # FAISS artifacts location (default updated)

def resolve_prefixes(slug: str = COUNCIL_KEY):
    docs = DOC_PREFIX.format(slug=slug).rstrip("/")
    idx  = INDEX_PREFIX.format(slug=slug).rstrip("/")
    return docs, idx

GCS_DOCS_PREFIX, GCS_INDEX_PREFIX = resolve_prefixes(COUNCIL_KEY)

# Logs under docs; meta sits with index
GCS_LOGS_PREFIX = f"{GCS_DOCS_PREFIX}/_logs"
GCS_META_BLOB   = f"{GCS_INDEX_PREFIX}/index_meta.json"

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
    for blob in bucket.list_blobs(prefix=prefix.rstrip("/") + "/"):
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
            bucket.blob(f"{gcs_prefix.rstrip('/')}/{rel}").upload_from_filename(str(p))

def _upload_bytes_to_gcs(bucket: storage.Bucket, dest_path: str, payload: bytes, content_type="application/pdf"):
    blob = bucket.blob(dest_path)
    blob.cache_control = "no-cache"
    blob.upload_from_string(payload, content_type=content_type)

def _read_index_meta(bucket: storage.Bucket) -> Optional[Dict[str, Any]]:
    # Prefer the script’s index_meta.json
    meta_blob = bucket.blob(GCS_META_BLOB)
    if meta_blob.exists():
        try:
            return json.loads(meta_blob.download_as_text())
        except Exception:
            pass
    # Backward-compat: read an old manifest if present
    legacy = bucket.blob(f"{GCS_INDEX_PREFIX}/manifest.json")
    if legacy.exists():
        try:
            return json.loads(legacy.download_as_text())
        except Exception:
            pass
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
    Simple PDF-only rebuild inside the app:
    Pull PDFs from GCS -> /tmp, build FAISS, push index back to GCS, write minimal meta.
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

    # Replace previous index path in GCS, then upload
    for b in list(bucket.list_blobs(prefix=f"{GCS_INDEX_PREFIX}/")):
        b.delete()
    _upload_dir_to_gcs(bucket, tmp_index, GCS_INDEX_PREFIX)

    dt = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        "generated_at_utc": dt,
        "council": COUNCIL_KEY,
        "chunks": len(split_docs),
        "builder": "app.py(pdf-only)",
        "index_prefix": GCS_INDEX_PREFIX,
        "docs_prefix": GCS_DOCS_PREFIX,
    }
    bucket.blob(GCS_META_BLOB).upload_from_string(json.dumps(meta, indent=2), content_type="application/json")
    log_event("reindex_pdf_only", meta)
    return meta

def rebuild_index_via_script(
    slug: str = COUNCIL_KEY,
    bucket: str = GCS_BUCKET,
    docs_prefix_tmpl: str = DOC_PREFIX,
    index_prefix_tmpl: str = INDEX_PREFIX,
    chunk_size: int = 1200,
    overlap: int = 180,
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    embed_batch: int = int(os.getenv("EMBED_BATCH", "64")),
) -> Dict[str, Any]:
    """
    Calls scripts/rebuild_index.py as a subprocess so we get multi-format support.
    """
    cmd = (
        f"python3 scripts/rebuild_index.py {shlex.quote(slug)} "
        f"--bucket {shlex.quote(bucket)} "
        f"--docs-prefix {shlex.quote(docs_prefix_tmpl)} "
        f"--index-prefix {shlex.quote(index_prefix_tmpl)} "
        f"--chunk-size {chunk_size} --overlap {overlap} "
        f"--embedding-model {shlex.quote(embedding_model)} "
        f"--embed-batch {embed_batch}"
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    ok = proc.returncode == 0
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    if not ok:
        raise RuntimeError(f"rebuild_index.py failed ({proc.returncode}).\nSTDOUT:\n{out}\n\nSTDERR:\n{err}")
    # After success, read fresh meta
    client = gcs_client(); bucket_obj = client.bucket(bucket)
    meta = _read_index_meta(bucket_obj) or {}
    meta["stdout"] = out[-4000:]  # keep tail for UI
    log_event("reindex_script", {"ok": True, **meta})
    return meta

def load_faiss_from_gcs() -> Optional[FAISS]:
    """
    Download FAISS index folder from GCS to /tmp/current_index and return a loaded FAISS store.
    """
    client = gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    tmp_index = Path("/tmp/current_index")
    _download_gcs_dir(bucket, f"{GCS_INDEX_PREFIX}/", tmp_index)

    faiss_bin = tmp_index / "index.faiss"
    faiss_pkl = tmp_index / "index.pkl"
    if not (faiss_bin.exists() and faiss_pkl.exists()):
        return None

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    return FAISS.load_local(str(tmp_index), embeddings, allow_dangerous_deserialization=True)

def upload_uploaded_file_to_kb(uploaded_file, subdir="uploads") -> str:
    """
    Save an uploaded file into gs://<bucket>/<DOC_PREFIX>/<subdir>/... and return its GCS path.
    """
    client = gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    safe_name = uploaded_file.name.replace("/", "_")
    ctype = uploaded_file.type or "application/octet-stream"
    dest = f"{GCS_DOCS_PREFIX}/{subdir}/{ts}-{safe_name}"
    _upload_bytes_to_gcs(bucket, dest, uploaded_file.getvalue(), content_type=ctype)
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

def _parse_source_from_chunk(txt: str) -> Optional[str]:
    # Expect the very first line to be like: [SOURCE] gs://bucket/path/to/file.ext
    first = (txt or "").splitlines()[:2]
    for line in first:
        m = re.match(r"\[SOURCE\]\s+(gs://\S+)", line.strip())
        if m:
            return m.group(1)
    return None

def render_sources(docs: List[Document]):
    if not docs:
        return
    with st.expander("Sources", expanded=False):
        for i, d in enumerate(docs, 1):
            src = getattr(d, "metadata", {}).get("source") if hasattr(d, "metadata") else None
            if not src:
                src = _parse_source_from_chunk(getattr(d, "page_content", ""))
            page = getattr(d, "metadata", {}).get("page") if hasattr(d, "metadata") else None
            pg = f" • p.{page+1}" if isinstance(page, int) else ""
            label = src or "unknown"
            st.markdown(f"{i}. `{label}`{pg}")

# -------------- UI: Header --------------
def header():
    client = gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    manifest = _read_index_meta(bucket)
    ts = (manifest or {}).get("generated_at_utc") or (manifest or {}).get("updated_at_utc") or "—"
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
    st.markdown("<hr style='opacity:0.2'/>", unsafe_allow_html=True)
    st.caption(
        f"Council: **{COUNCIL_KEY}**  •  Bucket: **{GCS_BUCKET}**  •  "
        f"Docs: **{GCS_DOCS_PREFIX}**  •  Index: **{GCS_INDEX_PREFIX}**  •  Data region: **{DATA_REGION}**"
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

            if st.button("🔄 Rebuild index now (PDF-only)"):
                with st.spinner("Rebuilding…"):
                    info = rebuild_index_from_gcs()
                st.toast("Index rebuilt")
                st.json(info)

            with st.expander("⚙️ Advanced rebuild (use script)", expanded=False):
                csize   = st.number_input("Chunk size", 200, 4000, 1200, 50)
                cover   = st.number_input("Chunk overlap", 0, 1000, 180, 10)
                emodel  = st.text_input("Embedding model", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
                ebatch  = st.number_input("Embed batch", 1, 512, int(os.getenv("EMBED_BATCH", "64")), 1)
                if st.button("Run scripts/rebuild_index.py"):
                    with st.spinner("Rebuilding via script… (multi-format)"):
                        try:
                            info = rebuild_index_via_script(
                                slug=COUNCIL_KEY,
                                bucket=GCS_BUCKET,
                                docs_prefix_tmpl=DOC_PREFIX,
                                index_prefix_tmpl=INDEX_PREFIX,
                                chunk_size=int(csize),
                                overlap=int(cover),
                                embedding_model=emodel,
                                embed_batch=int(ebatch),
                            )
                            st.success("Index rebuilt with script")
                            st.json(info)
                        except Exception as e:
                            st.error(str(e))

            if st.button("📜 Refresh logs"):
                pass

            # Display latest 200 logs
            try:
                client = gcs_client(); bucket = client.bucket(GCS_BUCKET)
                blobs = sorted(
                    [b for b in bucket.list_blobs(prefix=f"{GCS_LOGS_PREFIX}/")],
                    key=lambda b: b.name, reverse=True
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
    up = st.file_uploader("Upload a document", type=["pdf", "docx", "pptx", "xlsx", "xls", "csv", "rtf", "txt"])
    save_to_kb = st.checkbox("Save uploaded file to Knowledge Base (and reindex)", value=True)

    if "ephemeral_db" not in st.session_state:
        st.session_state["ephemeral_db"] = None
    if "ephemeral_retriever" not in st.session_state:
        st.session_state["ephemeral_retriever"] = None

    if up and st.button("Process file"):
        if save_to_kb:
            try:
                dest = upload_uploaded_file_to_kb(up)
                with st.spinner("Rebuilding index via script (multi-format)…"):
                    info = rebuild_index_via_script()
                st.success(f"Saved to GCS and reindexed.\nPath: gs://{GCS_BUCKET}/{dest}")
                st.caption(f"Index: gs://{GCS_BUCKET}/{GCS_INDEX_PREFIX}")
                st.json(info)
            except Exception as e:
                st.error(f"Could not save/reindex: {e}")
        else:
            # Ephemeral local index (session-only) for PDFs (fast path)
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
                st.success("Loaded PDF for this session. Ask away below.")
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
