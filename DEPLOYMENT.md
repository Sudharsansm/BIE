# BIE — Build, Push to GitHub, Deploy to PyPI

This guide takes the BIE project from this folder to a live GitHub repo
and a published PyPI package.

---

## 1. One-time setup

```bash
cd bie/

# Confirm package builds clean
pip install build twine
python -m build          # creates dist/*.whl and dist/*.tar.gz
twine check dist/*        # should print "PASSED" for both files
```

---

## 2. Push to GitHub

```bash
git init
git add .
git commit -m "feat: BitSearch Intelligence Engine v0.1.0 — Bitscrape-powered hybrid RAG API"

# Create the repo on GitHub first (via UI or `gh repo create`), then:
git remote add origin https://github.com/Sudharsansm/BitSearch-Intelligence-Engine.git
git branch -M main
git push -u origin main
```

---

## 3. Publish to PyPI

### Option A — Trusted Publisher (recommended, no secrets needed)

1. Go to https://pypi.org/manage/project/bitsearch-intelligence-engine/settings/publishing/
   (first create the project by uploading once manually, see Option B, OR
   pre-register the name via "Add a new pending publisher" before first release)
2. Add a Trusted Publisher:
   - Owner: `Sudharsansm`
   - Repository: `BitSearch-Intelligence-Engine`
   - Workflow: `ci.yml`
   - Environment: `pypi`
3. Tag and push a release — the included `.github/workflows/ci.yml` builds
   and publishes automatically:

```bash
git tag v0.1.0
git push origin v0.1.0
```

### Option B — Manual upload (first release / fallback)

```bash
python -m build
twine upload dist/*
# Username: __token__
# Password: <your PyPI API token>
```

---

## 4. Verify the published package

```bash
pip install bitsearch-intelligence-engine
bie serve --port 8000 &
bie search "AI chip demand 2026" --top-k 3
```

---

## 5. Running with a real LLM (optional)

BIE works out-of-the-box with extractive fallback answers. To get full
LLM-generated answers, point `BIE_LLM_BASE_URL` at any OpenAI-compatible
endpoint serving `sudharsansm/bie_qwen_2.5_3b`:

```bash
# Example: Ollama
ollama pull sudharsansm/bie_qwen_2.5_3b   # if published to Ollama library
export BIE_LLM_BASE_URL=http://localhost:11434/v1
export BIE_LLM_MODEL=sudharsansm/bie_qwen_2.5_3b
bie serve
```

---

## 6. Running with real embeddings (optional)

By default BIE tries `BAAI/bge-m3` via `sentence-transformers` and falls
back to a fast hash embedding if the model can't be downloaded (e.g. no
internet). For production-quality semantic search:

```bash
pip install "bitsearch-intelligence-engine[full]"
export BIE_EMBEDDING_DEVICE=cuda   # if GPU available
bie serve
```

---

## 7. Docker (optional)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ".[full]"
EXPOSE 8000
CMD ["bie", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t bie:0.1.0 .
docker run -p 8000:8000 \
  -e BIE_LLM_BASE_URL=http://host.docker.internal:11434/v1 \
  bie:0.1.0
```

---

## Versioning

Bump `version` in `pyproject.toml` and `bie/__init__.py`, commit, then
tag `vX.Y.Z` to trigger a new PyPI release via CI.
