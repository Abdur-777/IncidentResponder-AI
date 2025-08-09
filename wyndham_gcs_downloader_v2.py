#!/usr/bin/env python3
# Deep crawler for Wyndham: save locally + upload to GCS immediately (docs only by default)
import os, re, sys, time, hashlib, queue, io, csv
from pathlib import Path
from urllib.parse import urlparse, urljoin, urldefrag
import requests
from bs4 import BeautifulSoup

# ----- CONFIG (env-overridable) -----
START_URL   = os.getenv("START_URL", "https://www.wyndham.vic.gov.au/")
DOMAIN      = os.getenv("DOMAIN", "wyndham.vic.gov.au")
MAX_DEPTH   = int(os.getenv("MAX_DEPTH", "6"))        # crawl depth
MAX_PAGES   = int(os.getenv("MAX_PAGES", "8000"))     # safety limit
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "0.15")) # seconds between page fetches
TIMEOUT     = int(os.getenv("HTTP_TIMEOUT", "25"))
USER_AGENT  = os.getenv("USER_AGENT", "IncidentAI-Deep/1.1 (+https://incidentai.com)")
INCLUDE_IMAGES = os.getenv("INCLUDE_IMAGES", "false").lower() == "true"

ALLOWED_EXTS = (".pdf",".docx",".doc",".xlsx",".xls",".csv",".rtf",".pptx",".zip")
IMAGE_EXTS   = (".jpg",".jpeg",".png",".gif",".webp")

DOC_MIME_HINTS = (
    "application/pdf",
    "application/msword",
    "application/vnd.ms-excel",
    "application/rtf",
    "application/zip",
    "application/vnd.openxmlformats-officedocument",  # covers docx/xlsx/pptx
    "text/csv",
)

# Local save root
OUT_ROOT = Path("policies/wyndham")  # e.g. policies/wyndham/<category>/<file>

# ---------- Categories ----------
CATEGORY_RULES = [
    ("waste|bin|hard[-_ ]?rubbish|recycling|garbage|litter|landfill|compost|green waste|organic", "waste"),
    ("road|traffic|pothole|parking|transport|footpath|foot-path|bike|bicycle|bridge|intersection|roundabout", "roads"),
    ("animal|pet|dog|cat|livestock|impound|regist|wildlife|vet|microchip", "animals"),
    ("noise|nuisance|amenity|disturbance|music volume|loud", "noise"),
    ("rate|payment|valuation|fine|infringement|penalty|fees|charges|levy", "rates"),
    ("park|tree|environment|reserve|open[-_ ]space|green|garden|bushland|playground", "parks"),
    ("plan|permit|building|construction|overlay|zoning|town[-_ ]plan|development application|\\bDA\\b", "planning"),
    ("health|food|safety|public health|inspection|disease|mosquito|covid|hygiene", "health"),
    ("event|festival|workshop|class|program|community|volunteer|art|music|sport", "community"),
    ("youth|teen|young people|student|school holiday|mentoring", "youth"),
    ("senior|aged care|elderly|retirement|pensioner", "seniors"),
    ("library|book|reading|literacy|learning|study|tutor", "libraries"),
    ("emergency|disaster|flood|fire|bushfire|storm|evacuation", "emergency"),
    ("by[- ]?law|local law|governance|council meeting|minutes|policy", "governance"),
    ("business|permit|licen[sc]e|trading|market|startup", "business"),
    ("housing|homeless|accommodation|affordable housing", "housing"),
    ("child ?care|kindergarten|day care|family|parenting", "childcare"),
    ("sport|recreation|gym|fitness|swimming|leisure", "recreation"),
]
def guess_category(text: str) -> str:
    t = text.lower()
    for pat, cat in CATEGORY_RULES:
        if re.search(pat, t): return cat
    return "uncategorized"

# ---------- GCS ----------
GCS_BUCKET = os.getenv("GCS_BUCKET")
if not GCS_BUCKET:
    print("ERROR: set GCS_BUCKET env var (e.g., civreply-data).", file=sys.stderr); sys.exit(1)
try:
    from google.cloud import storage
    GCS_CLIENT = storage.Client()
    BUCKET = GCS_CLIENT.bucket(GCS_BUCKET)
except Exception as e:
    print("ERROR: google-cloud-storage not configured/authenticated.", file=sys.stderr)
    print("Hint: export GOOGLE_APPLICATIONS_CREDENTIALS=/path/to/service_account.json", file=sys.stderr)
    raise

BASE_PREFIX = "policies/wyndham"
HASH_INDEX  = f"{BASE_PREFIX}/_hash_index"
MANIFESTS   = f"{BASE_PREFIX}/manifests"

# ---------- HTTP ----------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
def http_get(url, **kw): return SESSION.get(url, timeout=TIMEOUT, **kw)

# ---------- Helpers ----------
def normalize(u: str) -> str:
    u, _ = urldefrag(u)
    return u.strip()

def is_internal(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == DOMAIN or host.endswith("." + DOMAIN)

def looks_like_doc(url: str) -> bool:
    path = urlparse(url).path.lower()

    # hard exclude images unless INCLUDE_IMAGES is true
    if not INCLUDE_IMAGES and any(path.endswith(ext) for ext in IMAGE_EXTS):
        return False

    # easy doc extensions
    if any(path.endswith(ext) for ext in ALLOWED_EXTS):
        return True

    # Drupal files often under /sites/default/files/ — detect by MIME
    if "/sites/default/files/" in path:
        try:
            r = http_get(url, stream=True)
            ctype = r.headers.get("content-type", "").lower()
            if not INCLUDE_IMAGES and ctype.startswith("image/"):
                return False
            return any(h in ctype for h in DOC_MIME_HINTS) or (INCLUDE_IMAGES and ctype.startswith("image/"))
        except Exception:
            return False

    return False

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def safe_filename(name: str) -> str:
    base = Path(name).stem
    ext  = Path(name).suffix
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)[:150]
    return (base or "file") + ext

def save_local_and_gcs(url: str, data: bytes):
    path = urlparse(url).path
    fname = safe_filename(Path(path).name or "file")
    cat = guess_category(fname + " " + url)

    # local
    dest_dir = OUT_ROOT / cat
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / fname
    with open(dest, "wb") as f:
        f.write(data)

    # GCS
    remote = f"{BASE_PREFIX}/{cat}/{fname}"
    BUCKET.blob(remote).upload_from_filename(str(dest))

    return cat, fname, dest, remote

def already_uploaded(h: str) -> bool:
    return BUCKET.blob(f"{HASH_INDEX}/{h}").exists()

def mark_uploaded(h: str):
    BUCKET.blob(f"{HASH_INDEX}/{h}").upload_from_string(b"")

def write_manifest(rows):
    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    path = f"{MANIFESTS}/docs-{ts}.csv"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["url","filename","category","size_bytes","sha256"])
    w.writerows(rows)
    BUCKET.blob(path).upload_from_string(buf.getvalue().encode("utf-8"), content_type="text/csv")
    print(f"[manifest] gs://{GCS_BUCKET}/{path}")

# ---------- Core ----------
def download_if_doc(url: str):
    try:
        r = http_get(url, stream=True); r.raise_for_status()
        data = r.content
        h = sha256_bytes(data)
        if already_uploaded(h):
            return ("skip", url, "dupe")
        cat, fname, dest, remote = save_local_and_gcs(url, data)
        mark_uploaded(h)
        print(f"[ok] {cat:<12} {len(data):>8} bytes  {fname}")
        return ("ok", url, cat, fname, len(data), h)
    except Exception as e:
        return ("err", url, str(e))

def crawl():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    seen = set()
    q = queue.Queue()
    q.put((normalize(START_URL), 0))

    pages_crawled = 0
    uploads = skips = errs = 0
    manifest_rows = []

    while not q.empty() and pages_crawled < MAX_PAGES:
        url, depth = q.get()
        if url in seen or depth > MAX_DEPTH:
            continue
        seen.add(url)

        try:
            r = http_get(url); r.raise_for_status()
            pages_crawled += 1
            if pages_crawled % 50 == 0:
                print(f"... crawled {pages_crawled} pages, queue={q.qsize()}, depth={depth}")
        except Exception as e:
            print(f"[warn] fetch failed {url} :: {e}", file=sys.stderr)
            continue

        soup = BeautifulSoup(r.text, "lxml")

        for a in soup.select("a[href]"):
            raw = a.get("href", "")
            link = normalize(urljoin(url, raw))
            if not link.startswith("http"): 
                continue
            if not is_internal(link): 
                continue

            if looks_like_doc(link):
                res = download_if_doc(link)
                if res[0] == "ok":
                    _, u, cat, fname, size, h = res
                    uploads += 1
                    manifest_rows.append([u, fname, cat, size, h])
                elif res[0] == "skip":
                    skips += 1
                else:
                    errs += 1
                    print(f"[err] {link} :: {res[2]}", file=sys.stderr)
            else:
                q.put((link, depth + 1))

        time.sleep(CRAWL_DELAY)

    if manifest_rows:
        write_manifest(manifest_rows)

    print(f"\nDone. pages={pages_crawled}, uploaded={uploads}, skipped={skips}, errors={errs}")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("WARNING: GOOGLE_APPLICATION_CREDENTIALS not set; GCS auth will fail.", file=sys.stderr)
    crawl()
