"""
BIE Quick-Start Example
========================
Shows how to use BIE end-to-end:
  1. Start the BIE server (in-process for the demo)
  2. Crawl a few URLs
  3. Run a hybrid search
  4. Run an agent (RAG) query

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import asyncio
import threading
import time

import uvicorn


async def main():
    from bie import BIEClient
    from bie.api import app, _index, _crawler, _trust
    from bie.models import DocumentRecord, ChunkRecord

    print("=" * 60)
    print("  BitSearch Intelligence Engine — Quick-Start Demo")
    print("=" * 60)

    # ── 1. Seed the index with some demo documents ────────────────────────
    print("\n📥  Seeding index with demo documents…")

    demo_docs = [
        {
            "url": "https://reuters.com/tech/tsmc-q2-2026",
            "title": "TSMC Q2 2026 Earnings",
            "text": (
                "TSMC reported record wafer shipments in Q2 2026, driven by surging AI chip demand. "
                "Revenue rose 38% year-over-year to $23.4 billion. CEO C.C. Wei said demand for "
                "advanced 3nm and 2nm nodes remains well above supply. Apple, NVIDIA, and AMD "
                "are among major customers expanding orders. TSMC plans $40 billion in new fab investment."
            ),
            "trust_score": 0.95,
        },
        {
            "url": "https://nature.com/articles/ai-alignment-2026",
            "title": "Advances in AI Alignment Research 2026",
            "text": (
                "A new wave of interpretability research has produced tools that can identify "
                "deceptive reasoning patterns in large language models. Researchers at several "
                "leading labs have demonstrated that constitutional AI methods reduce harmful "
                "outputs by over 70% on standard benchmarks. The field is rapidly maturing, "
                "with formal verification techniques being applied to smaller models."
            ),
            "trust_score": 0.97,
        },
        {
            "url": "https://bbc.com/news/semiconductor-shortage-2026",
            "title": "Global Semiconductor Supply Chain Update",
            "text": (
                "The global semiconductor supply chain is showing signs of normalisation in mid-2026. "
                "Automotive chip shortages that plagued manufacturers have largely resolved. "
                "However, advanced AI accelerator chips remain constrained. South Korea and Taiwan "
                "continue to dominate high-end logic production, while the US CHIPS Act has "
                "begun yielding results with Intel and TSMC Arizona fabs ramping production."
            ),
            "trust_score": 0.92,
        },
    ]

    from bie.crawler import TextChunker
    chunker = TextChunker(chunk_size=100)
    pairs = []
    for d in demo_docs:
        doc = DocumentRecord(
            url=d["url"],
            title=d["title"],
            text=d["text"],
            publish_date="2026-06-01",
            metadata={"site": d["url"].split("/")[2], "lang": "en",
                      "content_type": "article", "trust_score": d["trust_score"]},
        )
        chunks = chunker.chunk(doc)
        for c in chunks:
            c.trust_score = d["trust_score"]
        pairs.append((doc, chunks))

    count = await _index.add_documents(pairs)
    print(f"  ✅  Indexed {count} chunks across {len(pairs)} documents")

    # ── 2. Hybrid search ──────────────────────────────────────────────────
    print("\n🔍  Hybrid Search: 'semiconductor supply chain AI chips'\n")
    from bie.indexer import HybridRetriever
    retriever = HybridRetriever(_index)
    results = await retriever.search("semiconductor supply chain AI chips", top_k=3)

    for r in results:
        print(f"  [{r.rank}] {r.title}")
        print(f"       URL   : {r.url}")
        print(f"       Trust : {r.trust_score}  RRF: {r.rrf_score:.4f}")
        print(f"       Snippet: {r.snippet[:100]}…")
        print()

    # ── 3. Context Builder ────────────────────────────────────────────────
    print("📖  Building LLM context…")
    from bie.context import ContextBuilder
    cb = ContextBuilder()
    context, citations = cb.build(results, "semiconductor supply chain AI chips")
    print(f"  Context ({len(context)} chars, {len(citations)} citations):")
    print("  " + context[:300].replace("\n", "\n  ") + "…\n")

    # ── 4. Trust Engine ───────────────────────────────────────────────────
    print("🛡️   Trust Engine scores:")
    from bie.trust import TrustEngine
    te = TrustEngine()
    for url in ["https://reuters.com/news", "https://random-blog.xyz/post",
                "https://arxiv.org/abs/2406.12345", "https://whitehouse.gov/statement"]:
        print(f"  {url:<50}  → {te.score(url):.2f}")

    print("\n✅  BIE demo complete!")
    print("\nTo run the full API server:")
    print("  bie serve --host 0.0.0.0 --port 8000")
    print("\nThen query it:")
    print("  bie search 'TSMC earnings 2026' --top-k 3")
    print("  bie search 'AI alignment' --agent")


if __name__ == "__main__":
    asyncio.run(main())
