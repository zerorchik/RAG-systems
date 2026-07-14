# qdrant_vanille

Standalone experiments with the [Qdrant](https://qdrant.tech/) Python client — creating collections, upserting vectors, and querying — before combining Qdrant with an embedding model and an LLM in the other two projects in this repo.

## What's here

Basic Qdrant Cloud operations:

- Connecting to a Qdrant Cloud cluster
- Creating a collection with a given vector size and distance metric (cosine similarity)
- Upserting points (vectors + payload)
- Querying with `query_points` (top-k similarity search)

This project exists to isolate and understand the vector database layer on its own, separate from the embedding and generation logic used in [`try_text_RAG_system`](../try_text_RAG_system) and [`try_pdf_RAG_system`](../try_pdf_RAG_system).

## Setup

```bash
uv sync
cp .env.example .env
```

Fill in `.env`:

```
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_key
```

## Run

```bash
uv run main.py
```

## Note on the Qdrant client API

`qdrant-client` recently removed the older `search()` method in favor of `query_points()`. This project uses the current API — worth knowing if you find older Qdrant tutorials online that still reference `search()`.
