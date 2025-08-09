#!/usr/bin/env python3
import os, re, sys, io, csv, hashlib, concurrent.futures
from urllib.parse import urlparse, urljoin, urldefrag
import requests
from bs4 import BeautifulSoup
from google.cloud import storage

# -------- CONFIG --------
ROOT_SITEMAP = "https://www.wyndham.vic.gov.au/sitemap.xml"
ALLOWED_EXTS = (".pdf",".docx",".doc",".xlsx",".xls",".csv",".rtf",".pptx",".zip")
TIMEOUT      = 25
MAX_WORKERS  = 10
USER_AGENT   = "IncidentAI-Downloader/2.1 (+https://incidentai.com)"

# Wyndham host check
def is_wyndham(url):
    host = urlparse(url).netloc.lower()
    return host == "wyndham.vic.gov.au" or host.endswith(".wyndham.vic.gov.au")

# Categories
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
def guess_category(text):
    t = text.lower()
    for pat, cat in CATEGORY_RULES:
        if re.search(pat, t): return cat
    return "uncategorized"

# -------- GCS --------
BUCKET_NAME = os.getenv("GCS_BUCKET")
if not BUCKET_NAME:
    print("ERROR: set GCS_BUCKET", file=sys.stderr); sys.exit(1)
GCS = storage.Client()
BUCKET = GCS.bucket(BUCKET_NAME)
BASE_PREFIX    = "policies/wyndham"
HASH_INDEX_DIR = f"{BASE_PREFIX}/_hash_index"
MANIFEST_DIR   = f"{BASE_PREFIX}/manifests"

# -------- HTTP --------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
def http_get(url, **kw): return SESSION.get(url, timeout=TIMEOUT, **kw)

# -------- Helpers --------
def sha256_bytes(b):
    h=hashlib.sha256(); h.update(b); return h.hexdigest()

def blob_exists(name): return BUCKET.blob(name).exists()
def upload_bytes(name, data, ctype="application/octet-stream"):
    b=BUCKET.blob(name); b.upload_from_string(data, content_type=ctype); return b

def append_manifest(filename, rows, header):
    path = f"{MANIFEST_DIR}/{filename}"
    buf = io.StringIO()
    w = csv.writer(buf)
    if not BUCKET.blob(path).exists():
        w.writerow(header)
    else:
        buf.write(BUCKET.blob(path).download_as_text())
    for r in rows: w.writerow(r)
    upload_bytes(path, buf.getvalue().encode("utf-8"), "text/csv")
    return path

def normalize(u):
    u,_ = urldefrag(u)
    return u.strip()

def list_all_sitemap_urls(root):
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

def is_doc_link(url):
    path = urlparse(url).path.lower()
    if path.endswith(ALLOWED_EXTS): return True
    # Sometimes PDFs are served via ?download=1 etc — check MIME quickly
    try:
        head = http_get(url, stream=True)
        ctype = head.headers.get("content-type","").lower()
        return ("application/pdf" in ctype) or any(ext.strip(".") in ctype for ext in ALLOWED_EXTS)
    except Exception:
        return False

# -------- Core --------
def discover_doc_links_from_pages(pages):
    doc_links = set()
    def scan(page_url):
        try:
            r = http_get(page_url); r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            found = []
            for a in soup.select("a[href]"):
                link = normalize(urljoin(page_url, a["href"]))
                if not link.startswith("http"): continue
                if not is_wyndham(link): continue
                if is_doc_link(link):
                    found.append(link)
            return found
        except Exception:
            return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for links in ex.map(scan, pages):
            for l in links: doc_links.add(l)
    return sorted(doc_links)

def upload_doc(url):
    try:
        r = http_get(url); r.raise_for_status()
        data = r.content
        h = sha256_bytes(data)
        marker = f"{HASH_INDEX_DIR}/{h}"
        if blob_exists(marker): return ("skip", url)

        fname = os.path.basename(urlparse(url).path) or "file"
        cat   = guess_category(f"{fname} {url}")
        dest  = f"{BASE_PREFIX}/{cat}/{fname}"
        ctype = "application/pdf" if fname.lower().endswith(".pdf") else "application/octet-stream"
        upload_bytes(dest, data, ctype)
        upload_bytes(marker, b"")
        return ("ok", url, fname, cat, len(data), h)
    except Exception as e:
        return ("err", url, str(e))

def main():
    print("[*] Loading sitemap pages…")
    pages = list_all_sitemap_urls(ROOT_SITEMAP)
    html_pages = [u for u in pages if not any(urlparse(u).path.lower().endswith(ext) for ext in ALLOWED_EXTS)]
    print(f"[*] pages in sitemap={len(html_pages)}")

    print("[*] Scanning pages for document links…")
    doc_urls = discover_doc_links_from_pages(html_pages)
    print(f"[*] discovered doc URLs={len(doc_urls)}")

    ok=skip=err=0
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(upload_doc, doc_urls):
            if res[0]=="ok":
                _, url, fname, cat, size, h = res
                ok+=1; rows.append([url,fname,cat,size,h])
                print(f"[ok] {cat:<13} {size:>8}  {fname}")
            elif res[0]=="skip":
                skip+=1
            else:
                _, url, msg = res
                err+=1; print(f"[err] {url} :: {msg}", file=sys.stderr)

    manifest = append_manifest("docs.csv", rows, ["url","filename","category","size_bytes","sha256"])
    print(f"[manifest] gs://{BUCKET_NAME}/{manifest}")
    print(f"Done. uploaded={ok}, skipped={skip}, errors={err}")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("WARNING: GOOGLE_APPLICATION_CREDENTIALS not set", file=sys.stderr)
    main()
