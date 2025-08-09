#!/usr/bin/env python3
# Wyndham → GCS direct uploader (sitemap + on-page link discovery)
import os, re, sys, io, csv, hashlib, concurrent.futures
from urllib.parse import urlparse, urljoin, urldefrag
import requests
from bs4 import BeautifulSoup
from google.cloud import storage

# ---------- Required env ----------
BUCKET_NAME = os.getenv("GCS_BUCKET")
if not BUCKET_NAME:
    print("ERROR: set GCS_BUCKET env var", file=sys.stderr); sys.exit(1)

# ---------- Tunables (can override via env in the workflow) ----------
ROOT_SITEMAP = os.getenv("WyndhamSitemap", "https://www.wyndham.vic.gov.au/sitemap.xml")
MAX_WORKERS  = int(os.getenv("MAX_WORKERS", "10"))
TIMEOUT      = int(os.getenv("HTTP_TIMEOUT", "25"))
USER_AGENT   = os.getenv("USER_AGENT", "IncidentAI-Downloader/3.0 (+https://incidentai.com)")

ALLOWED_EXTS = (".pdf",".docx",".doc",".xlsx",".xls",".csv",".rtf",".pptx",".zip")

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
        if re.search(pat, t):
            return cat
    return "uncategorized"

# ---------- GCS ----------
GCS = storage.Client()
BUCKET = GCS.bucket(BUCKET_NAME)
BASE_PREFIX    = "policies/wyndham"
HASH_INDEX_DIR = f"{BASE_PREFIX}/_hash_index"
MANIFEST_PATH  = f"{BASE_PREFIX}/manifests/docs.csv"

def blob_exists(name: str) -> bool:
    return BUCKET.blob(name).exists()

def upload_bytes(name: str, data: bytes, ctype: str):
    BUCKET.blob(name).upload_from_string(data, content_type=ctype)

def append_manifest(rows):
    # append (or create with header)
    blob = BUCKET.blob(MANIFEST_PATH)
    buf = io.StringIO()
    w = csv.writer(buf)
    if not blob.exists():
        w.writerow(["url","filename","category","size_bytes","sha256"])
    else:
        buf.write(blob.download_as_text())
    for r in rows:
        w.writerow(r)
    upload_bytes(MANIFEST_PATH, buf.getvalue().encode("utf-8"), "text/csv")

# ---------- HTTP ----------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
def http_get(url, **kw): return SESSION.get(url, timeout=TIMEOUT, **kw)

def normalize(u: str) -> str:
    u, _ = urldefrag(u)
    return u.strip()

def is_wyndham(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == "wyndham.vic.gov.au" or host.endswith(".wyndham.vic.gov.au")

def list_sitemap_pages(root: str):
    urls, stack = set(), [root]
    while stack:
        sm = stack.pop()
        try:
            r = http_get(sm); r.raise_for_status()
        except Exception as e:
            print(f"[warn] sitemap fetch failed {sm}: {e}", file=sys.stderr); continue
        soup = BeautifulSoup(r.text, "xml")
        for s in soup.find_all("sitemap"):
            loc = s.find("loc");  loc and stack.append(loc.get_text(strip=True))
        for u in soup.find_all("url"):
            loc = u.find("loc");  loc and urls.add(normalize(loc.get_text(strip=True)))
    return [u for u in urls if is_wyndham(u)]

def looks_like_doc(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in ALLOWED_EXTS)

def is_doc_link(url: str) -> bool:
    # quick pass by extension; if uncertain, peek headers
    if looks_like_doc(url): return True
    try:
        r = http_get(url, stream=True)
        ctype = r.headers.get("content-type","").lower()
        return ("application/pdf" in ctype) or any(ext.strip(".") in ctype for ext in ALLOWED_EXTS)
    except Exception:
        return False

def discover_doc_links(pages):
    found = set()
    def scan(page_url):
        try:
            r = http_get(page_url); r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            links = []
            for a in soup.select("a[href]"):
                link = normalize(urljoin(page_url, a["href"]))
                if not link.startswith("http"): continue
                if not is_wyndham(link): continue
                if is_doc_link(link):
                    links.append(link)
            return links
        except Exception:
            return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for links in ex.map(scan, pages):
            for l in links: found.add(l)
    return sorted(found)

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def upload_doc(url: str):
    try:
        r = http_get(url); r.raise_for_status()
        data = r.content
        h = sha256_bytes(data)
        marker = f"{HASH_INDEX_DIR}/{h}"
        if blob_exists(marker):
            return ("skip", url)

        fname = os.path.basename(urlparse(url).path) or "file"
        cat   = guess_category(f"{fname} {url}")
        dest  = f"{BASE_PREFIX}/{cat}/{fname}"
        ctype = "application/pdf" if fname.lower().endswith(".pdf") else "application/octet-stream"

        upload_bytes(dest, data, ctype)
        upload_bytes(marker, b"", "application/octet-stream")
        return ("ok", url, fname, cat, len(data), h)
    except Exception as e:
        return ("err", url, str(e))

def main():
    print("[*] Loading sitemap pages…")
    pages = list_sitemap_pages(ROOT_SITEMAP)
    pages = [u for u in pages if not looks_like_doc(u)]
    print(f"[*] pages in sitemap={len(pages)}")

    print("[*] Discovering document links on pages…")
    doc_urls = discover_doc_links(pages)
    print(f"[*] discovered doc URLs={len(doc_urls)}")

    ok = skip = err = 0
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(upload_doc, doc_urls):
            if res[0] == "ok":
                _, url, fname, cat, size, h = res
                ok += 1; rows.append([url, fname, cat, size, h])
                print(f"[ok] {cat:<13} {size:>8}  {fname}")
            elif res[0] == "skip":
                skip += 1
            else:
                _, url, msg = res
                err += 1; print(f"[err] {url} :: {msg}", file=sys.stderr)

    if rows:
        append_manifest(rows)
        print(f"[manifest] gs://{BUCKET_NAME}/{MANIFEST_PATH}")
    print(f"Done. uploaded={ok}, skipped={skip}, errors={err}")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("WARNING: GOOGLE_APPLICATION_CREDENTIALS is not set (won't auth in local runs).", file=sys.stderr)
    main()
