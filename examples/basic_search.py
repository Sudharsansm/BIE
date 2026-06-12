"""
Basic BIE usage example.

Run with:
    python examples/basic_search.py
"""

import bie


def main() -> None:
    # One-shot: crawl a couple of pages and search them
    results = bie.search(
        query="python web scraping",
        urls=["https://quotes.toscrape.com/"],
        top_k=5,
        max_pages=5,
        max_depth=0,  # don't follow links — avoids bitscrape scheduler tie-break bug
        use_embeddings=False,
    )

    for i, r in enumerate(results, 1):
        print(f"{i}. {r.title}")
        print(f"   {r.url}")
        print(f"   score={r.score:.4f}")
        print(f"   {r.snippet}")
        print()


if __name__ == "__main__":
    main()
