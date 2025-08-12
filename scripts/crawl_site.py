#!/usr/bin/env python3
"""
Crawl a council website and push docs to GCS.

Example:
  python scripts/crawl_site.py wyndham \
    --base https://www.wyndham.vic.gov.au \
    --bucket civreply-data \
    --docs-prefix policies/{slug}/crawl \
    --max-pages 2000 --delay 0.5 --file-exts .pdf

Notes:
- Defaults save under gs://<bucket>/policies/<slug>/crawl/...
- Only internal links are followed.
- By default grabs PDFs; you can include other types via --file-exts
"""

import os, time, re, hashlib, argparse
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from google.cloud import storage
import urllib.robotparser as robotparser

DEFAULT_FILE_EXTS = (".pdf",)  # add more if wanted, e.g. ".docx", ".xlsx", ".pptx", ".csv"

def is_internal(href: str, base_netloc: str) -> bool:
    p = urlparse(href)
    return (p.netloc == "" or p.netloc == base_netloc)

def allowed_by_robots(base: str, user_agent: str) -> robotparser.RobotFileParser:
    rp = robotparser.RobotFileParser()
    robots_url = urljoin(base, "/robots.txt")
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        pass
    return rp

def save_blob(bucket: storage.Bucket, path: str, content: bytes, content_type: str):
    blob = bucket.blob(path)
    blob.cache_control = "no-cache"
    blob.upload_from_string(content, content_type=content_type)

def main():
    parser = argparse.ArgumentParser(description="Crawl a site and push docs to GCS")
    parser.add_argument("slug", help="Council slug, e.g., wyndham")
    parser.add_argument("--base", required=True, help="Base URL to crawl, e.g., https://www.wyndham.vic.gov.au")
    parser.add_argument("--bucket", default=os.getenv("GCS_BUCKET", "civreply-data"))
    parser.add_argument("--docs-prefix", default=os.getenv("DOCS_PREFIX", "policies/{slug}/crawl"))
    parser.add_argument("--max-pages", type=int, default=int(os.getenv("CRAWL_MAX_PAGES", "2000")))
    parser.add_argument("--delay", type=float, default=float(os.getenv("CRAWL_DELAY", "0.5")))
    parser.add_argument("--file-exts", default=",".join(DEFAULT_FILE_EXTS),
                        help="Comma-separated list like .pdf,.docx")
    args = parser.parse_args()

    args.docs_prefix = args.docs_prefix.format(slug=args.slug).rstrip("/")

    file_exts = tuple([e.strip() for e in args.file_exts.split(",") if e.strip().startswith(".")])
    base = args.base.rstrip("/")
    domain = urlparse(base).netloc

    session = requests.Session()
    session.headers.update({"User-Agent": "CivReplyCrawler/1.0 (+github-actions)"})

    rp = allowed_by_robots(base, session.headers["User-Agent"])

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    seen, q = set(), [base]
    saved = 0
    pages = 0

    print(f"→ Crawling {base} (max_pages={args.max_pages}, delay={args.delay}s, file_exts={file_exts})")
    while q and pages < args.max_pages:
        url = q.pop(0)
        if url in seen:
            continue
        seen.add(url)

        if rp and not rp.can_fetch(session.headers["User-Agent"], url):
            continue

        try:
            resp = session.get(url, timeout=15)
        except Exception as e:
            continue
        if not resp.ok or "text/html" not in resp.headers.get("content-type",""):
            continue

        pages += 1
        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            if not is_internal(href, domain):
                continue

            lower = href.lower()
            if lower.endswith(file_exts):
                # Download and push to GCS
                try:
                    r = session.get(href, timeout=30)
                    if not r.ok:
                        continue
                    ctype = r.headers.get("content-type", "application/octet-stream").split(";")[0]
                    name = href.split("/")[-1].split("?")[0]
                    # prefix with a short hash for uniqueness
                    h = hashlib.sha1(href.encode()).hexdigest()[:10]
                    key = f"{args.docs_prefix}/{h}-{name}"
                    save_blob(bucket, key, r.content, ctype)
                    saved += 1
                    if saved % 10 == 0:
                        print(f"   …saved {saved} files")
                except Exception:
                    continue
            else:
                # Queue next HTML page if it belongs to the same site
                if href.startswith(base) and href not in seen:
                    q.append(href)

        time.sleep(args.delay)

    print(f"✓ Crawl complete. Pages visited: {pages}, Files saved: {saved}")
    print(f"   Output prefix: gs://{args.bucket}/{args.docs_prefix}/")

if __name__ == "__main__":
    main()
