"""
BIE CLI — bie serve / bie search / bie crawl
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="bie",
        description="BitSearch Intelligence Engine CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # ── serve ──────────────────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Start the BIE API server")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true")
    serve_p.add_argument("--workers", type=int, default=1)

    # ── search ──────────────────────────────────────────────────────────────
    search_p = sub.add_parser("search", help="Run a search query against a running BIE instance")
    search_p.add_argument("query")
    search_p.add_argument("--url", default="http://localhost:8000")
    search_p.add_argument("--api-key", default="dev-key")
    search_p.add_argument("--top-k", type=int, default=5)
    search_p.add_argument("--agent", action="store_true", help="Use agent (LLM answer) mode")

    # ── crawl ───────────────────────────────────────────────────────────────
    crawl_p = sub.add_parser("crawl", help="Crawl and index a URL")
    crawl_p.add_argument("url")
    crawl_p.add_argument("--server", default="http://localhost:8000")
    crawl_p.add_argument("--api-key", default="dev-key")

    args = parser.parse_args()

    if args.command == "serve":
        _serve(args)
    elif args.command == "search":
        asyncio.run(_search(args))
    elif args.command == "crawl":
        asyncio.run(_crawl(args))
    else:
        parser.print_help()
        sys.exit(0)


def _serve(args):
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    uvicorn.run(
        "bie.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


async def _search(args):
    from bie.client import BIEClient

    async with BIEClient(base_url=args.url, api_key=args.api_key) as client:
        if args.agent:
            print(f"\n🤖  Agent query: {args.query!r}\n")
            resp = await client.agent_query(args.query, top_k=args.top_k)
            print(resp.answer)
            print(f"\n📚  Citations ({len(resp.citations)}):")
            for c in resp.citations:
                print(f"  [{c.index}] {c.title} — {c.url}")
        else:
            print(f"\n🔍  Search: {args.query!r}\n")
            resp = await client.search(args.query, top_k=args.top_k)
            print(f"Found {resp.total_found} results in {resp.latency_ms:.0f} ms\n")
            for r in resp.results:
                print(
                    f"  [{r.rank}] {r.title}\n"
                    f"       {r.url}\n"
                    f"       trust={r.trust_score}  rrf={r.rrf_score}\n"
                    f"       {r.snippet[:120]}…\n"
                )


async def _crawl(args):
    from bie.client import BIEClient

    async with BIEClient(base_url=args.server, api_key=args.api_key) as client:
        print(f"Crawling {args.url} …")
        resp = await client.crawl_url(args.url)
        print(f"Status: {resp.status} — {resp.message}")


if __name__ == "__main__":
    main()
