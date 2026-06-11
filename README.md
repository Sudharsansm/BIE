# BIE — BitSearch Intelligence Engine

[![PyPI](https://img.shields.io/pypi/v/bie.svg)](https://pypi.org/project/bie/)
[![Python](https://img.shields.io/pypi/pyversions/bie.svg)](https://pypi.org/project/bie/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Built on Bitscrape](https://img.shields.io/badge/built%20on-Bitscrape-orange.svg)](https://github.com/Sudharsansm/Bitscrape)

**The fastest, simplest way to give any LLM, RAG pipeline, or AI agent
real-time, citation-backed web search and extraction.**

BIE crawls the live web (powered by [**Bitscrape**](https://pypi.org/project/bitscrape/),
our high-performance async crawler), builds a hybrid **BM25 + semantic
vector** index in memory, and returns ranked, source-attributed results —
all from a single Python call, REST endpoint, CLI command, or
[MCP](https://modelcontextprotocol.io) tool.

```python
import bie

results = bie.search(
    "latest semiconductor export rules 2026",
    urls=["https://www.reuters.com/technology/"],
)

for r in results:
    print(r.title, "—", r.url, f"(score={r.score:.3f})")
```

---

## Why BIE?

- 🚀 **Zero infra** — no Elasticsearch, no Milvus, no Kafka. Pure Python,
  in-memory hybrid index. Scale up later if you need to.
- 🧠 **Hybrid retrieval out of the box** — BM25 lexical search fused with
  sentence-transformer embeddings via Reciprocal Rank Fusion.
- 🤖 **MCP-ready** — drop-in tool for Claude Desktop, Claude Code, and any
  MCP-compatible AI app.
- ⚡ **Powered by Bitscrape** — async, polite (robots.txt-aware), and fast
  crawling/extraction under the hood.
- 🔌 **Use anywhere** — Python library, REST API, CLI, or MCP server.

---

## Install

```bash
pip install bits-bie
```

> Note: the PyPI **distribution** is named `bits-bie` (since `bie`
> was too similar to an existing PyPI project), but you still
> `import bie` and run the `bie` CLI command — same API as shown below.

Optional extras:

```bash
pip install "bits-bie[embeddings]"  # semantic/vector search (sentence-transformers)
pip install "bits-bie[server]"      # FastAPI + Uvicorn REST server
pip install "bits-bie[mcp]"         # Model Context Protocol server
pip install "bits-bie[all]"         # everything
```

> BIE depends on [`bitscrape`](https://pypi.org/project/bitscrape/), our
> proprietary async crawling & extraction framework, which is installed
> automatically.

---

## Usage

### 1. One-shot search (Python)

```python
import bie

results = bie.search("AI regulation news", urls=["https://example.com/news"], top_k=5)
for r in results:
    print(r)
```

### 2. Build a reusable index

```python
from bie import BIE

engine = BIE()
engine.crawl(["https://example.com/blog", "https://another-site.com"])

print(engine.search("quarterly earnings"))
print(engine.search("product launch"))  # reuses the same index
```

### 3. Index your own text (no crawling)

```python
engine.add_text(
    url="internal://doc-1",
    title="Q2 Strategy Memo",
    text="...",
    trust_score=1.0,
)
```

### 4. CLI

```bash
# Crawl + search in one command
bie search "global markets today" --url https://www.bbc.com/news --top-k 5

# Just crawl & dump extracted pages
bie crawl https://example.com --max-pages 20 --out docs.jsonl

# Run the REST API
bie serve --port 8000

# Run as an MCP server (stdio)
bie mcp
```

### 5. REST API

```bash
bie serve --port 8000
```

```bash
curl -X POST http://localhost:8000/crawl/url \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com/news"]}'

curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "latest news", "top_k": 5}'
```

See the full endpoint contract in [`docs/API.md`](docs/API.md).

### 6. MCP (Model Context Protocol)

Add BIE as a tool in your MCP client (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "bie": {
      "command": "bie",
      "args": ["mcp"]
    }
  }
}
```

This exposes three tools to your AI assistant:

- `bie_search(query, urls, top_k, max_pages)` — crawl + search in one call
- `bie_crawl(urls, max_pages)` — crawl & index into a session-persistent store
- `bie_index_search(query, top_k)` — search the session index

---

## Configuration

All settings can be set via environment variables prefixed with `BIE_`,
or passed directly:

```python
from bie import BIE, BIESettings

engine = BIE(BIESettings(
    max_pages=20,
    max_depth=1,
    use_embeddings=True,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    bm25_weight=0.6,
    vector_weight=0.4,
))
```

| Setting | Env var | Default | Description |
|---|---|---|---|
| `max_pages` | `BIE_MAX_PAGES` | `40` | Max pages crawled per seed URL |
| `max_depth` | `BIE_MAX_DEPTH` | `2` | Max link-follow depth |
| `concurrent_requests` | `BIE_CONCURRENT_REQUESTS` | `16` | Crawl concurrency |
| `robotstxt_obey` | `BIE_ROBOTSTXT_OBEY` | `true` | Respect robots.txt |
| `use_embeddings` | `BIE_USE_EMBEDDINGS` | `true` | Enable semantic search |
| `chunk_size` | `BIE_CHUNK_SIZE` | `800` | Chars per chunk |
| `bm25_weight` / `vector_weight` | `BIE_BM25_WEIGHT` / `BIE_VECTOR_WEIGHT` | `0.5` / `0.5` | Fusion weights |
| `api_key` | `BIE_API_KEY` | `None` | If set, requires `Authorization: Bearer <key>` |

---

## Architecture

```
              ┌─────────────────────────────────────────┐
              │                  bie                     │
              │                                           │
   urls ──▶   │  Crawler (Bitscrape)                     │
              │     │                                     │
              │     ▼                                     │
              │  Document → Chunker → HybridIndex         │
              │                         │   │             │
              │                  BM25Index  VectorIndex   │
              │                         │   │             │
              │                       Fusion (RRF)        │
              │                         │                 │
   query ──▶  │                         ▼                 │
              │                  Ranked SearchResults      │
              └─────────────────────────────────────────┘
                     │            │            │
                  Python API   REST API    MCP Server
```

This OSS edition implements the core of the BIE PRD's **Module 1
(Crawler)**, **Module 2 (Indexes)**, **Module 3 (Hybrid Retriever)**, and
**Module 11 (Agent API)** as a single lightweight package — no external
services required. Larger deployments can swap `BM25Index`/`VectorIndex`
for Elasticsearch/Milvus-backed implementations behind the same
`HybridIndex` interface.

---

## Built on Bitscrape

BIE's crawling and extraction layer is powered by
[**Bitscrape**](https://github.com/Sudharsansm/Bitscrape)
(`pip install bitscrape`), our async, robots.txt-aware web scraping
framework — giving BIE high-performance, polite, production-grade crawling
out of the box.

---

## License

MIT — see [LICENSE](LICENSE).
