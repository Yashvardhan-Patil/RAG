"""A documented simple pipeline over an existing Chroma collection."""

import time
from pathlib import Path

import chromadb

from rag_eval_studio.models import GoldenCase, Observation, ProviderConfig
from rag_eval_studio.providers import embed_query, generate

SYSTEM_PROMPT = (
    "Answer the question using the supplied context. If the context is insufficient, "
    "say that you do not know. Do not invent supporting facts."
)


def run_existing_chroma(
    case: GoldenCase,
    *,
    chroma_path: str | Path,
    collection_name: str,
    embedding: ProviderConfig,
    generation: ProviderConfig,
    top_k: int,
) -> Observation:
    """Retrieve, generate once, and capture timings and provider metadata."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    path = Path(chroma_path)
    if not path.is_dir():
        raise ValueError("Chroma path must be an existing directory")
    if not (path / "chroma.sqlite3").is_file():
        raise ValueError("Chroma path contains no persistent collection database")
    if not collection_name.strip():
        raise ValueError("collection name is required")

    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_collection(name=collection_name, embedding_function=None)
    started = time.perf_counter()
    embedded = embed_query(case.question, embedding)
    result = collection.query(
        query_embeddings=[embedded.vector], n_results=top_k, include=["documents"]
    )
    retrieved_at = time.perf_counter()
    ids = tuple(result["ids"][0])
    documents = result.get("documents")
    context = (
        tuple(documents[0])
        if documents and documents[0] and all(isinstance(x, str) for x in documents[0])
        else (() if not ids else None)
    )

    parts = (
        [f"[{doc_id}] {text}" for doc_id, text in zip(ids, context, strict=True)]
        if context is not None
        else []
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{'\n\n'.join(parts)}\n\nQuestion: {case.question}",
        },
    ]
    generated = generate(messages, generation)
    finished = time.perf_counter()
    return Observation(
        response=generated.text,
        retrieved_contexts=context,
        retrieved_doc_ids=ids,
        retrieval_latency_ms=(retrieved_at - started) * 1000,
        generation_latency_ms=(finished - retrieved_at) * 1000,
        total_latency_ms=(finished - started) * 1000,
        embedding_tokens=embedded.tokens,
        generation_prompt_tokens=generated.prompt_tokens,
        generation_completion_tokens=generated.completion_tokens,
        retrieval_cost_usd=embedded.cost_usd,
        generation_cost_usd=generated.cost_usd,
    )
