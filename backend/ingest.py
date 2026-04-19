import os
import time
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

SEED_URL = "https://docs.stripe.com"
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
CRAWLED_URLS_FILE = "crawled_urls.txt"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
REQUEST_DELAY = 0.5
SKIP_EXTENSIONS = {".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".xml", ".json"}

# text-embedding-3-small pricing: $0.02 per 1M tokens
COST_PER_TOKEN = 0.02 / 1_000_000


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.scheme.startswith("http"):
        return False
    if not parsed.netloc.endswith("docs.stripe.com"):
        return False
    if parsed.fragment and not parsed.path:
        return False
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    return True


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="", query="").geturl().rstrip("/")


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.select("nav, footer, header, script, style, aside, [class*='sidebar'], [class*='nav'], [id*='nav'], [id*='sidebar'], [class*='menu']"):
        tag.decompose()

    main = soup.select_one("main, article, [role='main'], .content, #content, .docs-content")
    target = main if main else soup.body
    if not target:
        return ""

    text = target.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def crawl(seed: str) -> list[dict]:
    visited: set[str] = set()
    queue: list[str] = [normalize_url(seed)]
    pages: list[dict] = []

    session = requests.Session()
    session.headers.update({"User-Agent": "stripe-api-scout/1.0 (docs ingest)"})

    while queue:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            response = session.get(url, timeout=15)
        except requests.RequestException as e:
            print(f"  ERROR fetching {url}: {e}")
            time.sleep(REQUEST_DELAY)
            continue

        if response.status_code != 200:
            print(f"  SKIP [{response.status_code}]: {url}")
            time.sleep(REQUEST_DELAY)
            continue

        print(f"Crawling [{len(visited)}]: {url}")

        text = extract_text(response.text)
        if text:
            pages.append({"url": url, "text": text})

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all("a", href=True):
            try:
                href = tag["href"]
                absolute = urljoin(url, href)
                normalized = normalize_url(absolute)
                if normalized not in visited and normalized not in queue and is_valid_url(normalized) and "/changelog/" not in normalized:
                    queue.append(normalized)
            except Exception:
                continue

        time.sleep(REQUEST_DELAY)

    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[dict] = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            chunks.append({"text": split, "metadata": {"source": page["url"]}})
    return chunks


def estimate_cost(chunks: list[dict]) -> float:
    total_chars = sum(len(c["text"]) for c in chunks)
    estimated_tokens = total_chars / 4
    return estimated_tokens * COST_PER_TOKEN


def embed_and_store(chunks: list[dict]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    BATCH_SIZE = 100
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = embeddings.embed_documents(texts)
        records = [
            {
                "id": f"chunk-{batch_start + i}",
                "values": vectors[i],
                "metadata": {
                    "text": batch[i]["text"],
                    "source": batch[i]["metadata"].get("source", ""),
                },
            }
            for i in range(len(batch))
        ]
        index.upsert(vectors=records)
        if (batch_start // BATCH_SIZE) % 100 == 0:
            print(f"  Upserted {batch_start + len(batch)}/{len(chunks)} chunks")


def main() -> None:
    print("=== Stripe Docs Ingest ===\n")

    pages = crawl(SEED_URL)

    crawled_urls = [p["url"] for p in pages]
    with open(CRAWLED_URLS_FILE, "w") as f:
        f.write("\n".join(crawled_urls))
    print(f"\nSaved {len(crawled_urls)} crawled URLs to {CRAWLED_URLS_FILE}")

    print("\nChunking content...")
    chunks = chunk_pages(pages)

    cost = estimate_cost(chunks)
    print(f"\nTotal URLs crawled : {len(pages)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Estimated embedding cost: ${cost:.4f} (text-embedding-3-small @ $0.02/1M tokens)")

    print("\nEmbedding and storing in Pinecone...")
    embed_and_store(chunks)
    print("Done. Vectors upserted to Pinecone.")


if __name__ == "__main__":
    main()
