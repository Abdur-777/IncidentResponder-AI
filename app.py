# =========================
#  RENDER GCS AUTH SETUP
# =========================
import os, json, io, csv, shutil, tempfile, datetime
from pathlib import Path

if "GCP_SA_JSON" in os.environ:
    with open("/tmp/gcs-key.json", "w") as f:
        f.write(os.environ["GCP_SA_JSON"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcs-key.json"

# =========================
#  IMPORTS
# =========================
import streamlit as st
from dotenv import load_dotenv
from google.cloud import storage

# file parsers
from PyPDF2 import PdfReader
import docx  # python-docx
from pptx import Presentation
import pandas as pd
from striprtf.striprtf import rtf_to_text

# langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# =========================
#  ENV
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "civreply-data")
COUNCIL        = os.getenv("COUNCIL", "wyndham")
DOC_PREFIX     = os.getenv("DOC_PREFIX", f"policies/{COUNCIL}/")
INDEX_PREFIX   = os.getenv("INDEX_PREFIX", f"indexes/{COUNCIL}/faiss")
LOCAL_INDEX    = Path(f"index/{COUNCIL}")
INDEX_ON_START = os.getenv("INDEX_ON_START", "false").lower() == "true"

PUBLIC_MODE    = os.getenv("PUBLIC_MODE", "false").lower() == "true"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

BRAND_LOGO_URL = os.getenv("BRAND_LOGO_URL", "")
BRAND_PRIMARY  = os.getenv("BRAND_PRIMARY", "#0055a5")

if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY")
    st.stop()

SUPPORTED_EXTS = (".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv", ".rtf", ".txt")

# =========================
#  PAGE
# =========================
st.set_page_config(page_title="IncidentResponder AI", page_icon="🚨", layout="wide")

# --- Public banner ---
if PUBLIC_MODE:
    st.markdown(
        """
        <div style="background:#fff6d6;border:1px solid #f0d78a;border-radius:10px;padding:8px 12px;margin:6px 0;">
          <b>Public mode:</b> read-only access to the Wyndham Knowledge Base. Uploads & admin tools are disabled.
        </div>
        """, unsafe_allow_html=True
    )

# Branding bar + last indexed
def gcs_client():
    return storage.Client()

def latest_index_time_str():
    try:
        blobs = list(gcs_client().list_blobs(GCS_BUCKET, prefix=f"{INDEX_PREFIX}/"))
        if not blobs:
            return "N/A"
        newest = max((b.updated for b in blobs if b.updated), default=None)
        if newest:
            return newest.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass
    return "N/A"

with st.container():
    col1, col2 = st.columns([6, 4])
    with col1:
        if BRAND_LOGO_URL:
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:14px;margin-top:6px;">
                    <img src="{BRAND_LOGO_URL}" style="height:42px"/>
                    <div style="font-size:26px;font-weight:800;color:{BRAND_PRIMARY};">
                        {COUNCIL.title()} — IncidentResponder AI
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='font-size:28px;font-weight:800;color:{BRAND_PRIMARY};margin:6px 0'>"
                f"{COUNCIL.title()} — IncidentResponder AI</div>",
                unsafe_allow_html=True
            )
        st.caption("Answers from council policies & documents. Always up to date.")
    with col2:
        st.markdown(
            f"""
            <div style="text-align:right;margin-top:4px;">
                <div style="font-size:13px;color:#666;">Last indexed</div>
                <div style="font-size:15px;font-weight:600;">{latest_index_time_str()}</div>
            </div>
            """, unsafe_allow_html=True
        )

st.divider()

# =========================
#  GCS HELPERS
# =========================
def list_supported_blobs():
    client = gcs_client()
    blobs = list(client.list_blobs(GCS_BUCKET, prefix=DOC_PREFIX))
    files = [b for b in blobs if b.name.lower().endswith(SUPPORTED_EXTS)]
    files.sort(key=lambda b: b.updated or 0, reverse=True)
    return files

def download_dir_from_gcs(prefix: str, dest_dir: Path) -> bool:
    client = gcs_client()
    blobs = list(client.list_blobs(GCS_BUCKET, prefix=prefix.rstrip("/") + "/"))
    if not blobs:
        return False
    tmp = Path(tempfile.mkdtemp())
    for b in blobs:
        rel = b.name[len(prefix)+1:]
        if not rel:
            continue
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        b.download_to_filename(str(p))
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in tmp.rglob("*"):
        if p.is_file():
            tgt = dest_dir / p.relative_to(tmp)
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(tgt))
    return True

def upload_dir_to_gcs(src_dir: Path, prefix: str):
    client = gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    for p in src_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src_dir).as_posix()
            dest = f"{prefix.rstrip('/')}/{rel}"
            bucket.blob(dest).upload_from_filename(str(p))

def signed_url_for_blob_path(gcs_path: str, minutes: int = 120) -> str:
    if gcs_path.startswith("gs://"):
        _, rest = gcs_path.split("gs://", 1)
        bucket_name, key = rest.split("/", 1)
    else:
        bucket_name, key = GCS_BUCKET, gcs_path
    try:
        bucket = gcs_client().bucket(bucket_name)
        blob = bucket.blob(key)
        return blob.generate_signed_url(
            expiration=datetime.timedelta(minutes=minutes),
            method="GET"
        )
    except Exception:
        return ""

# =========================
#  TEXT EXTRACTORS
# =========================
def _join_lines(lines): return "\n".join([str(line).strip() for line in lines if str(line).strip()])
def extract_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        return _join_lines(page.extract_text() or "" for page in reader.pages)
    except Exception: return ""
def extract_docx(data: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(data))
        return _join_lines(p.text for p in doc.paragraphs)
    except Exception: return ""
def extract_pptx(data: bytes) -> str:
    try:
        prs = Presentation(io.BytesIO(data))
        slides = []
        for s in prs.slides:
            texts = []
            for shp in s.shapes:
                if hasattr(shp, "text"): texts.append(shp.text)
            slides.append(_join_lines(texts))
        return _join_lines(slides)
    except Exception: return ""
def extract_xlsx(data: bytes) -> str:
    try:
        with pd.ExcelFile(io.BytesIO(data)) as xls:
            frames = []
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                frames.append(f"[Sheet: {sheet}]\n" + df.to_string(index=False))
            return "\n\n".join(frames)
    except Exception: return ""
def extract_xls(data: bytes) -> str:
    try:
        df = pd.read_excel(io.BytesIO(data), engine="xlrd", sheet_name=None)
        chunks = []
        for sheet, sdf in df.items():
            chunks.append(f"[Sheet: {sheet}]\n" + sdf.to_string(index=False))
        return "\n\n".join(chunks)
    except Exception: return ""
def extract_csv(data: bytes) -> str:
    try:
        for enc in ("utf-8","utf-8-sig","latin-1"):
            try: txt = data.decode(enc, errors="ignore"); break
            except Exception: continue
        out = []
        for row in csv.reader(io.StringIO(txt)):
            out.append(", ".join(row))
        return _join_lines(out)
    except Exception: return ""
def extract_rtf(data: bytes) -> str:
    try:
        txt = data.decode("utf-8", errors="ignore")
        return rtf_to_text(txt)
    except Exception: return ""
def extract_txt(data: bytes) -> str:
    for enc in ("utf-8","utf-8-sig","latin-1"):
        try: return data.decode(enc, errors="ignore")
        except Exception: continue
    return ""
EXTRACTORS = {
    ".pdf": extract_pdf, ".docx": extract_docx, ".pptx": extract_pptx,
    ".xlsx": extract_xlsx, ".xls": extract_xls, ".csv": extract_csv,
    ".rtf": extract_rtf, ".txt": extract_txt,
}
def extract_any(name: str, data: bytes) -> str:
    fn = EXTRACTORS.get(Path(name.lower()).suffix)
    return fn(data) if fn else ""

# =========================
#  INDEX BUILD / LOAD
# =========================
def list_supported_blobs():  # re-declare to avoid forward ref issues
    client = gcs_client()
    blobs = list(client.list_blobs(GCS_BUCKET, prefix=DOC_PREFIX))
    files = [b for b in blobs if b.name.lower().endswith(tuple(EXTRACTORS.keys()))]
    files.sort(key=lambda b: b.updated or 0, reverse=True)
    return files

def build_index_from_gcs() -> bool:
    st.session_state["build_logs"] = []
    blobs = list_supported_blobs()
    if not blobs:
        st.session_state["build_logs"].append(f"No files under gs://{GCS_BUCKET}/{DOC_PREFIX}")
        return False

    texts = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)
    st.session_state["build_logs"].append(f"Indexing {len(blobs)} files from gs://{GCS_BUCKET}/{DOC_PREFIX}")

    for i, b in enumerate(blobs, 1):
        try:
            data = b.download_as_bytes()
        except Exception as e:
            st.session_state["build_logs"].append(f"[skip] {b.name} download failed: {e}")
            continue

        body = extract_any(b.name, data)
        if not body.strip():
            st.session_state["build_logs"].append(f"[skip] {b.name} had no extractable text")
            continue

        header = f"[SOURCE] gs://{GCS_BUCKET}/{b.name}\n"
        for chunk in splitter.split_text(header + body):
            texts.append(chunk)

        if i % 20 == 0:
            st.session_state["build_logs"].append(f"... parsed {i} files")

    if not texts:
        st.session_state["build_logs"].append("No text chunks produced.")
        return False

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vs = FAISS.from_texts(texts, embeddings)
    LOCAL_INDEX.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(LOCAL_INDEX))
    upload_dir_to_gcs(LOCAL_INDEX, INDEX_PREFIX)

    # write metadata
    meta = {
        "council": COUNCIL,
        "chunks": len(texts),
        "generated_at_utc": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        gcs_client().bucket(GCS_BUCKET).blob(f"{INDEX_PREFIX}/index_meta.json")\
            .upload_from_string(json.dumps(meta, indent=2).encode("utf-8"),
                                content_type="application/json")
    except Exception:
        pass

    st.session_state["build_logs"].append(f"[done] Index uploaded to gs://{GCS_BUCKET}/{INDEX_PREFIX}")
    return True

def load_vectorstore():
    if LOCAL_INDEX.exists():
        try:
            emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            return FAISS.load_local(str(LOCAL_INDEX), emb, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"Local FAISS present but failed to load: {e}")

    if download_dir_from_gcs(INDEX_PREFIX, LOCAL_INDEX):
        try:
            emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            return FAISS.load_local(str(LOCAL_INDEX), emb, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"GCS FAISS downloaded but failed to load: {e}")

    if INDEX_ON_START:
        with st.spinner("Building index from GCS (first boot can take a few minutes)…"):
            ok = build_index_from_gcs()
        if ok:
            emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            return FAISS.load_local(str(LOCAL_INDEX), emb, allow_dangerous_deserialization=True)

    return None

def get_llm():
    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

# =========================
#  QUERY LOGGING (CSV in GCS)
# =========================
def log_query_to_gcs(mode: str, question: str, answer: str, sources: list):
    try:
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        day = datetime.datetime.utcnow().strftime("%Y%m%d")
        key = f"logs/{COUNCIL}/queries-{day}.csv"

        bucket = gcs_client().bucket(GCS_BUCKET)
        blob = bucket.blob(key)

        src_str = " | ".join(sources)[:2000] if sources else ""
        # safely quote fields
        def q(x): return '"' + str(x).replace('"', "'") + '"'
        line = f"{q(ts)},{q(mode)},{q(question)},{q(answer[:2000])},{q(src_str)}\n"

        if blob.exists():
            existing = blob.download_as_text()
            updated = existing + line
        else:
            header = "timestamp_utc,mode,question,answer_preview,sources\n"
            updated = header + line

        blob.upload_from_string(updated, content_type="text/csv")
    except Exception:
        pass  # logging is best-effort

# =========================
#  SIDEBAR: ADMIN (hidden in public)
# =========================
st.sidebar.title("Menu")
is_admin = st.session_state.get("is_admin", False)

if not PUBLIC_MODE:
    if not is_admin:
        pwd = st.sidebar.text_input("Admin password", type="password")
        if st.sidebar.button("Login") and ADMIN_PASSWORD and pwd == ADMIN_PASSWORD:
            st.session_state["is_admin"] = True
            is_admin = True
            st.sidebar.success("Admin unlocked.")
    if is_admin:
        st.sidebar.markdown("### Admin")
        if st.sidebar.button("Rebuild Index Now"):
            with st.spinner("Rebuilding index…"):
                ok = build_index_from_gcs()
            st.sidebar.success("Rebuild finished." if ok else "No files or build failed.")
        # Today’s log link (signed)
        day = datetime.datetime.utcnow().strftime("%Y%m%d")
        log_key = f"logs/{COUNCIL}/queries-{day}.csv"
        if gcs_client().bucket(GCS_BUCKET).blob(log_key).exists():
            href = signed_url_for_blob_path(f"gs://{GCS_BUCKET}/{log_key}", minutes=60)
            st.sidebar.markdown(f"[Download today’s log]({href})")
else:
    st.sidebar.info("Public mode is on (admin hidden).")

# =========================
#  UI MODES
# =========================
if PUBLIC_MODE:
    mode = "Wyndham Knowledge Base (GCS)"
else:
    mode = st.radio("Choose a data source", ["Wyndham Knowledge Base (GCS)", "Upload a File"], horizontal=True)

kb = None
if mode.startswith("Wyndham"):
    kb = load_vectorstore()
    if not kb:
        st.info(
            "No FAISS index available yet. "
            "Set INDEX_ON_START=true to build on boot, or run your nightly index job.\n\n"
            f"Docs:  gs://{GCS_BUCKET}/{DOC_PREFIX}\n"
            f"Index: gs://{GCS_BUCKET}/{INDEX_PREFIX}"
        )

uploaded = None
if mode == "Upload a File" and not PUBLIC_MODE:
    uploaded = st.file_uploader("📄 Upload a document", type=[e.lstrip(".") for e in SUPPORTED_EXTS])

q = st.text_input("💬 Ask a question:")

# =========================
#  ANSWER + “View Policy” buttons
# =========================
def parse_sources_from_docs(docs):
    out, seen = [], set()
    for d in docs:
        first = (d.page_content.splitlines() or [""])[0].strip()
        if first.startswith("[SOURCE]"):
            s = first.replace("[SOURCE]", "").strip()
            if s not in seen:
                out.append(s); seen.add(s)
    return out

def show_extracts_with_links(docs):
    with st.expander("📄 Relevant Extracts (with sources)"):
        for i, d in enumerate(docs, 1):
            src = "Unknown"
            first = (d.page_content.splitlines() or [""])[0].strip()
            if first.startswith("[SOURCE]"):
                src = first.replace("[SOURCE]", "").strip()
            st.markdown(f"**Extract {i}** — _{src}_")
            st.write(d.page_content)
            if src.startswith("gs://"):
                signed = signed_url_for_blob_path(src, minutes=120)
                if signed:
                    st.link_button("View Policy", signed, use_container_width=False, key=f"lnk_{i}_{hash(src)%10**6}")
            st.markdown("---")

if q:
    if mode.startswith("Wyndham"):
        if not kb:
            st.error("Knowledge base not loaded yet.")
        else:
            with st.spinner("🔎 Searching Wyndham documents…"):
                retriever = kb.as_retriever(search_kwargs={"k": 6})
                docs = retriever.get_relevant_documents(q)
                context = "\n\n".join(d.page_content for d in docs)
                prompt = f"""Answer on behalf of {COUNCIL.title()} City Council using ONLY the context.
If answer is unclear/not covered, say so and suggest the next step (correct council link, phone, or form).
Return a concise answer (4–8 sentences).

[CONTEXT]
{context}

[QUESTION]
{q}
"""
                ans = get_llm().predict(prompt)

            st.subheader("📌 AI Response")
            st.write(ans)
            show_extracts_with_links(docs)
            log_query_to_gcs("kb", q, ans, parse_sources_from_docs(docs))

    else:
        if PUBLIC_MODE:
            st.error("Uploads are disabled in Public mode.")
        elif not uploaded:
            st.error("Please upload a document first.")
        else:
            name = uploaded.name
            data = uploaded.read()
            body = extract_any(name, data)
            if not body.strip():
                st.error("Couldn’t extract text from that file.")
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(f"[SOURCE] (uploaded) {name}\n" + body)
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vs = FAISS.from_texts(chunks, embeddings)

                with st.spinner("📑 Analyzing your document…"):
                    docs = vs.as_retriever(search_kwargs={"k": 6}).get_relevant_documents(q)
                    context = "\n\n".join(d.page_content for d in docs)
                    prompt = f"""Use ONLY the uploaded document context to answer succinctly.
If unsure, say so and suggest a next step. Prefer including a relevant link mentioned if available.

[CONTEXT]
{context}

[QUESTION]
{q}
"""
                    ans = get_llm().predict(prompt)

                st.subheader("📌 AI Response")
                st.write(ans)
                show_extracts_with_links(docs)
                log_query_to_gcs("upload", q, ans, parse_sources_from_docs(docs))

# =========================
#  FOOTER / LOGS
# =========================
if not PUBLIC_MODE:
    with st.expander("🛠 Index build logs"):
        for line in st.session_state.get("build_logs", []):
            st.text(line)

st.markdown(
    f"""
    <hr/>
    <div style='color:#7a7a7a;font-size:12px'>
      Bucket: <code>{GCS_BUCKET}</code> • Council: <code>{COUNCIL}</code> • Docs: <code>{DOC_PREFIX}</code> • Index: <code>{INDEX_PREFIX}</code>
      <br/>© {datetime.datetime.utcnow().year} IncidentResponder AI
    </div>
    """,
    unsafe_allow_html=True
)
