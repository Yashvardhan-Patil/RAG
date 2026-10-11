import chromadb
import pytest

from rag_eval_studio import pipeline
from rag_eval_studio.dataset import parse_case
from rag_eval_studio.models import ProviderConfig
from rag_eval_studio.pipeline import run_existing_chroma
from rag_eval_studio.providers import EmbeddingResult, GenerationResult


def test_existing_chroma_queries_ranked_ids_and_generates_once(tmp_path, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("docs")
    collection.add(
        ids=["a", "b"],
        documents=["alpha", "beta"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    prompts = []
    monkeypatch.setattr(
        pipeline,
        "embed_query",
        lambda *_: EmbeddingResult([1.0, 0.0], 2, 0.001),
    )

    def fake_generate(messages, _config):
        prompts.append(messages)
        return GenerationResult("alpha answer", 8, 2, 0.002)

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    config = ProviderConfig("openai", "test")
    case = parse_case({"question": "Which?", "reference_answer": "alpha"})
    observed = run_existing_chroma(
        case,
        chroma_path=tmp_path,
        collection_name="docs",
        embedding=config,
        generation=config,
        top_k=1,
    )
    assert observed.retrieved_doc_ids == ("a",)
    assert observed.retrieved_contexts == ("alpha",)
    assert observed.response == "alpha answer"
    assert "[a] alpha" in prompts[0][1]["content"]
    assert observed.total_latency_ms >= observed.retrieval_latency_ms >= 0
    assert collection.count() == 2


def test_missing_collection_fails_before_embedding(tmp_path, monkeypatch):
    called = False

    def fake_embed(*_):
        nonlocal called
        called = True

    monkeypatch.setattr(pipeline, "embed_query", fake_embed)
    config = ProviderConfig("openai", "test")
    case = parse_case({"question": "Q", "reference_answer": "A"})
    with pytest.raises(ValueError, match="no persistent collection database"):
        run_existing_chroma(
            case,
            chroma_path=tmp_path,
            collection_name="missing",
            embedding=config,
            generation=config,
            top_k=1,
        )
    assert not called
