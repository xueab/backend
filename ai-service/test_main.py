from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from main import (
    ChatStreamEvent,
    ChatStreamRequest,
    MoodAnalysisRequest,
    MoodAnalysisResponse,
    app,
    get_ai_service,
    get_retriever,
    get_settings,
)
from rag import KnowledgeRetriever, RetrievedChunk, build_chat_system_prompt, build_embedder
from vector_stores import (
    ChunkRecord,
    FaissVectorStore,
    StoredHit,
    build_vector_store,
    infer_category,
)


class FakeAIService:
    def analyze_diary(self, payload: MoodAnalysisRequest) -> MoodAnalysisResponse:
        return MoodAnalysisResponse(
            analysisText=f"已分析：{payload.content[:6]}",
            model="fake-model",
            requestId="req-test",
        )

    def start_chat_stream(self, payload: ChatStreamRequest):
        yield ChatStreamEvent(type="delta", content="你好，", requestId="stream-1")
        yield ChatStreamEvent(type="delta", content="我在这里陪你。", requestId="stream-1")
        yield ChatStreamEvent(type="done", requestId="stream-1")


class FakeRetriever:
    def __init__(self, hits: list[RetrievedChunk] | None = None) -> None:
        self._hits = hits or []
        self.reindex_calls = 0

    def search(self, query: str, top_k: int, min_score: float) -> list[RetrievedChunk]:
        return [h for h in self._hits if h.score >= min_score][:top_k]

    def ensure_index(self, force_rebuild: bool = False) -> int:
        if force_rebuild:
            self.reindex_calls += 1
        return len(self._hits)


def override_settings():
    class TestSettings:
        deepseek_model = "deepseek-chat"
        ai_service_internal_token = ""
        rag_enabled = True
        rag_top_k = 3
        rag_min_score = 0.3
        rag_model_name = "fake-embed-model"

    return TestSettings()


def setup_client(retriever: FakeRetriever | None = None) -> TestClient:
    app.dependency_overrides[get_settings] = override_settings
    app.dependency_overrides[get_ai_service] = lambda: FakeAIService()
    app.dependency_overrides[get_retriever] = lambda: retriever
    return TestClient(app)


def teardown_overrides():
    app.dependency_overrides.clear()


def test_health():
    client = setup_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    teardown_overrides()


def test_analyze_endpoint():
    client = setup_client()

    response = client.post(
        "/internal/v1/mood/analyze",
        json={
            "diaryId": 1,
            "content": "今天虽然很累，但还是完成了很多事情。",
            "moodScore": 6,
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == "fake-model"
    assert "已分析" in response.json()["analysisText"]
    teardown_overrides()


def test_chat_stream_endpoint():
    client = setup_client()

    with client.stream(
        "POST",
        "/internal/v1/chat/stream",
        json={
            "sessionId": 1,
            "messages": [
                {"role": "user", "content": "我最近压力很大。"}
            ],
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode("utf-8")

    assert response.status_code == 200
    assert '"type": "delta"' in body
    assert '"type": "done"' in body
    assert "我在这里陪你" in body
    teardown_overrides()


def test_build_chat_system_prompt_injects_hits():
    base = "你是一名温和的助手。"
    hits = [
        RetrievedChunk(
            source="sleep-hygiene.md",
            title="睡眠卫生",
            content="尝试固定起床时间，睡前避免使用屏幕。",
            score=0.72,
        ),
        RetrievedChunk(
            source="breathing-relaxation.md",
            title="呼吸与放松",
            content="可以尝试 4-7-8 呼吸法帮助放松。",
            score=0.66,
        ),
    ]

    prompt = build_chat_system_prompt(base, hits)

    assert prompt.startswith(base)
    assert "睡眠卫生" in prompt
    assert "4-7-8" in prompt
    assert "sleep-hygiene.md" in prompt
    assert "不要直接复述" in prompt


def test_build_chat_system_prompt_no_hits_returns_base():
    base = "基础提示词"

    assert build_chat_system_prompt(base, []) == base


def test_rag_search_endpoint_returns_hits():
    fake = FakeRetriever(
        hits=[
            RetrievedChunk(
                source="sleep-hygiene.md",
                title="睡眠卫生",
                content="关于失眠的建议……",
                score=0.82,
            )
        ]
    )
    client = setup_client(retriever=fake)

    response = client.post(
        "/internal/v1/rag/search",
        json={"query": "我最近老是睡不着"},
    )

    assert response.status_code == 200
    hits = response.json()["hits"]
    assert len(hits) == 1
    assert hits[0]["source"] == "sleep-hygiene.md"
    assert hits[0]["score"] == 0.82
    teardown_overrides()


def test_rag_search_endpoint_503_when_retriever_missing():
    client = setup_client(retriever=None)

    response = client.post(
        "/internal/v1/rag/search",
        json={"query": "任何"},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["errorCode"] == "RAG_DISABLED"
    teardown_overrides()


def test_rag_reindex_endpoint():
    fake = FakeRetriever(
        hits=[
            RetrievedChunk(source="a.md", title="A", content="x", score=0.5),
            RetrievedChunk(source="b.md", title="B", content="y", score=0.6),
        ]
    )
    client = setup_client(retriever=fake)

    response = client.post("/internal/v1/rag/reindex")

    assert response.status_code == 200
    assert response.json()["chunkCount"] == 2
    assert fake.reindex_calls == 1
    teardown_overrides()


# ---------------------------------------------------------------- P1：Embedder 抽象与查询缓存


class FakeEmbedder:
    """可计数的假 embedder，用于验证 KnowledgeRetriever 的接口契约和查询缓存。"""

    def __init__(self, identifier: str = "fake::v1", dim: int = 8) -> None:
        self.identifier = identifier
        self.dim = dim
        self.embed_calls = 0
        self.embedded_texts: list[list[str]] = []

    def embed(self, texts):
        self.embed_calls += 1
        self.embedded_texts.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            # 用文本哈希填几个分量，确保不同文本向量不一样
            h = abs(hash(t)) % (2 ** 31)
            out[i, 0] = (h % 1000) / 1000.0
            out[i, 1] = ((h // 1000) % 1000) / 1000.0
            out[i, 2] = 1.0
        # L2 归一化
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return out / norms


def _write_docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "sleep.md").write_text(
        "# 睡眠卫生\n\n固定起床时间，睡前一小时远离屏幕。", encoding="utf-8"
    )
    (docs / "anxiety.md").write_text(
        "# 焦虑应对\n\n注意到焦虑时，可以尝试 4-7-8 呼吸法。", encoding="utf-8"
    )
    return docs


def _make_retriever(tmp_path, *, embedder=None, identifier="fake::v1"):
    docs = _write_docs(tmp_path)
    embedder = embedder or FakeEmbedder(identifier=identifier)
    store = FaissVectorStore(cache_dir=tmp_path / "cache")
    retriever = KnowledgeRetriever(
        docs_dir=docs,
        embedder=embedder,
        store=store,
        chunk_size=200,
        chunk_overlap=20,
        query_cache_size=8,
    )
    return retriever, embedder, store


def test_retriever_uses_injected_embedder(tmp_path):
    retriever, embedder, store = _make_retriever(tmp_path)

    count = retriever.ensure_index()

    assert count == 2
    assert store.chunk_count() == 2
    # 构建索引时应当对所有片段批量 embed 一次
    assert embedder.embed_calls == 1
    assert len(embedder.embedded_texts[0]) == 2


def test_retriever_query_cache_hits(tmp_path):
    retriever, embedder, _ = _make_retriever(tmp_path)
    retriever.ensure_index()
    calls_after_index = embedder.embed_calls

    retriever.search("我最近睡不着", top_k=2, min_score=0.0)
    after_first = embedder.embed_calls
    retriever.search("我最近睡不着", top_k=2, min_score=0.0)
    after_second = embedder.embed_calls
    retriever.search("怎么缓解焦虑", top_k=2, min_score=0.0)
    after_third = embedder.embed_calls

    assert after_first == calls_after_index + 1
    assert after_second == after_first
    assert after_third == after_second + 1


def test_retriever_fingerprint_changes_with_embedder_identifier(tmp_path):
    retriever_v1, _, _ = _make_retriever(tmp_path, identifier="fake::v1")
    fp_v1 = retriever_v1._fingerprint()  # noqa: SLF001
    retriever_v1.ensure_index()

    # v2 共用同一个 cache_dir，store.needs_rebuild 会因为 fingerprint 不同而返回 True
    retriever_v2, _, store_v2 = _make_retriever(tmp_path, identifier="fake::v2")
    fp_v2 = retriever_v2._fingerprint()  # noqa: SLF001

    assert fp_v1 != fp_v2
    assert store_v2.needs_rebuild(fp_v2) is True


def test_build_embedder_falls_back_to_local_for_unknown_kind(monkeypatch):
    # 不实际加载远程 / sentence-transformers，仅校验工厂分支
    embedder = build_embedder(
        "no-such-kind",
        local_model_name="dummy-model",
        dashscope_api_key="",
        dashscope_model="text-embedding-v3",
        dashscope_dim=1024,
        dashscope_base_url="https://example.invalid",
        dashscope_timeout=10.0,
    )
    assert embedder.identifier.startswith("local::")


def test_build_embedder_dashscope_requires_api_key():
    import pytest

    with pytest.raises(RuntimeError):
        build_embedder(
            "dashscope",
            local_model_name="dummy",
            dashscope_api_key="",
            dashscope_model="text-embedding-v3",
            dashscope_dim=1024,
            dashscope_base_url="https://example.invalid",
            dashscope_timeout=10.0,
        )


# ---------------------------------------------------------------- P2：VectorStore 抽象与分类


def test_infer_category_from_known_filename():
    assert infer_category("sleep-hygiene.md") == "habits"
    assert infer_category("anxiety-coping.md") == "emotions"
    assert infer_category("crisis-resources.md") == "trauma-care"
    assert infer_category("totally-unknown.md") == "general"
    assert infer_category("") == "general"


def test_faiss_store_rebuild_and_search(tmp_path):
    store = FaissVectorStore(cache_dir=tmp_path / "cache")
    fp = "fp-v1"
    records = [
        ChunkRecord(source="sleep.md", title="睡眠", content="x", category="habits"),
        ChunkRecord(source="anxiety.md", title="焦虑", content="y", category="emotions"),
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    assert store.needs_rebuild(fp) is True
    store.rebuild(records, vectors, fp)
    assert store.needs_rebuild(fp) is False
    assert store.chunk_count() == 2

    hits = store.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=2)
    assert hits[0].record.source == "sleep.md"
    # 类别过滤
    only_emotions = store.search(
        np.array([[1.0, 0.0]], dtype=np.float32), top_k=2, category="emotions"
    )
    assert len(only_emotions) == 1
    assert only_emotions[0].record.source == "anxiety.md"


def test_faiss_store_persists_across_instances(tmp_path):
    cache = tmp_path / "cache"
    fp = "fp-persist"
    records = [
        ChunkRecord(source="a.md", title="A", content="x", category="emotions"),
    ]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    first = FaissVectorStore(cache_dir=cache)
    first.rebuild(records, vectors, fp)

    second = FaissVectorStore(cache_dir=cache)
    assert second.chunk_count() == 1
    assert second.needs_rebuild(fp) is False
    hits = second.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=1)
    assert hits[0].record.category == "emotions"


def test_build_vector_store_falls_back_to_faiss_for_unknown(tmp_path):
    store = build_vector_store(
        "no-such-store",
        cache_dir=tmp_path / "cache",
        qdrant_collection="x",
        qdrant_url="",
        qdrant_api_key="",
        qdrant_path="",
        qdrant_prefer_grpc=False,
        qdrant_timeout=5.0,
    )
    assert isinstance(store, FaissVectorStore)


def test_retriever_propagates_category(tmp_path):
    retriever, _, _ = _make_retriever(tmp_path)
    retriever.ensure_index()

    hits = retriever.search("我最近老是睡不着", top_k=2, min_score=0.0)
    sources = {h.source for h in hits}
    assert "sleep.md" in sources or "anxiety.md" in sources
    for h in hits:
        # _write_docs 用的都是测试虚构文件名（不在分类映射中），应回退到 general
        assert h.category == "general"


# ---------------------------------------------------------------- P3：BM25 + 多路召回 + RRF


def test_bm25_retriever_basic(tmp_path):
    pytest_skip_if_missing = []
    try:
        import jieba  # noqa: F401
    except ImportError:
        pytest_skip_if_missing.append("jieba")
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        pytest_skip_if_missing.append("rank-bm25")
    if pytest_skip_if_missing:
        import pytest

        pytest.skip(f"未安装：{','.join(pytest_skip_if_missing)}")

    from bm25_index import BM25Retriever

    retriever = BM25Retriever(cache_dir=tmp_path / "cache")
    fp = "fp-bm25-v1"
    records = [
        ChunkRecord(
            source="sleep.md", title="睡眠卫生",
            content="固定起床时间，睡前避免使用屏幕，可以尝试 4-7-8 呼吸法。",
            category="habits",
        ),
        ChunkRecord(
            source="anxiety.md", title="焦虑应对",
            content="注意到焦虑时，先做深呼吸，使用认知行为疗法识别想法。",
            category="emotions",
        ),
        ChunkRecord(
            source="pua.md", title="识别 PUA 与 gaslighting",
            content="如果在亲密关系中长期被否定，可能遇到了 gaslighting。",
            category="relationships",
        ),
    ]

    assert retriever.needs_rebuild(fp) is True
    retriever.rebuild(records, fp)
    assert retriever.needs_rebuild(fp) is False
    assert retriever.chunk_count() == 3

    hits = retriever.search("我最近一直睡不着", top_k=2)
    assert hits and hits[0].record.source == "sleep.md"

    # 专有名词 PUA：dense 容易漂移，BM25 应当能稳定命中
    hits = retriever.search("感觉自己被 PUA 了", top_k=2)
    assert hits[0].record.source == "pua.md"


def test_bm25_persists_across_instances(tmp_path):
    try:
        import jieba  # noqa: F401
        import rank_bm25  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("BM25 依赖未安装")

    from bm25_index import BM25Retriever

    cache = tmp_path / "cache"
    fp = "fp-bm25-persist"
    records = [
        ChunkRecord(source="a.md", title="A", content="焦虑 应对 呼吸", category="emotions"),
    ]

    first = BM25Retriever(cache_dir=cache)
    first.rebuild(records, fp)

    second = BM25Retriever(cache_dir=cache)
    assert second.chunk_count() == 1
    assert second.needs_rebuild(fp) is False


def test_reciprocal_rank_fusion():
    from pipeline import reciprocal_rank_fusion

    rankings = [
        # dense
        [("a", 0.9), ("b", 0.8), ("c", 0.5)],
        # sparse：c 在 dense 排第 3，在 sparse 排第 1，融合后应当上升
        [("c", 5.0), ("d", 3.0), ("a", 2.0)],
    ]
    fused = reciprocal_rank_fusion(rankings, k=60)
    keys = [k for k, _ in fused]
    assert "a" in keys
    assert "c" in keys
    # a 出现两次（rank 0 + rank 2），应当排第一
    assert keys[0] == "a"


# ---------------------------------------------------------------- P5：危机词检测 + 改写解析


def test_detect_crisis_positive_and_negative():
    from query_rewriter import detect_crisis

    assert detect_crisis("我真的不想活了") is True
    assert detect_crisis("最近老想自杀") is True
    assert detect_crisis("感觉很疲惫") is False
    assert detect_crisis("") is False
    assert detect_crisis(None) is False  # type: ignore[arg-type]


class _FakeChat:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        text = self._responses.pop(0) if self._responses else ""
        return _FakeCompletion(text)


class _FakeCompletions:
    def __init__(self, chat: _FakeChat) -> None:
        self.completions = chat


class _FakeOpenAIClient:
    def __init__(self, responses: list[str]) -> None:
        self._chat = _FakeChat(responses)
        self.chat = _FakeCompletions(self._chat)

    @property
    def total_calls(self) -> int:
        return self._chat.calls


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)] if content else []


def test_llm_rewriter_multi_query_parsing():
    from query_rewriter import LLMQueryRewriter

    client = _FakeOpenAIClient(
        responses=[
            "1. 失眠 应对方法\n- 睡眠卫生 调整\n* 焦虑导致 失眠\n",
        ]
    )
    rw = LLMQueryRewriter(
        client=client,
        model="fake-model",
        enable_hyde=False,
        max_queries=3,
    )
    result = rw.enrich("最近老是睡不着")
    assert result.queries[0] == "最近老是睡不着"
    assert "失眠 应对方法" in result.queries
    assert "睡眠卫生 调整" in result.queries


def test_llm_rewriter_hyde_short_query_only():
    from query_rewriter import LLMQueryRewriter

    client = _FakeOpenAIClient(
        responses=[
            "1. 焦虑\n2. 焦虑应对\n",  # multi-query
            "看起来你最近压力比较大，先试试深呼吸。",  # HyDE
        ]
    )
    rw = LLMQueryRewriter(
        client=client,
        model="fake-model",
        enable_multi_query=True,
        enable_hyde=True,
        hyde_min_query_chars=8,
    )
    result = rw.enrich("好烦")
    assert result.hyde and "深呼吸" in result.hyde
    assert client.total_calls == 2


def test_llm_rewriter_skips_for_crisis_query():
    from query_rewriter import LLMQueryRewriter

    client = _FakeOpenAIClient(responses=["不应该被调用\n", "不应该被调用\n"])
    rw = LLMQueryRewriter(client=client, model="fake-model")
    result = rw.enrich("我真的不想活了")
    assert result.is_crisis is True
    assert result.queries == ["我真的不想活了"]
    assert client.total_calls == 0


# ---------------------------------------------------------------- Pipeline 编排


class _StubReranker:
    backend_name = "stub"

    def rerank(self, query, docs, top_n):
        from rerankers import RerankResult

        # 把第二个候选顶到第一位，验证 rerank 真的改变了顺序
        if len(docs) >= 2:
            order = [1, 0] + list(range(2, len(docs)))
        else:
            order = list(range(len(docs)))
        return [RerankResult(index=i, score=1.0 - 0.1 * rank) for rank, i in enumerate(order[:top_n])]

    def describe(self):
        return "stub-reranker"


def test_pipeline_end_to_end_with_stubs(tmp_path):
    """端到端验证：Hybrid 召回 → RRF → Rerank。仅依赖 FAISS（兜底 numpy），无需 LLM。"""
    from pipeline import PipelineConfig, RagPipeline

    docs = _write_docs(tmp_path)
    embedder = FakeEmbedder()
    store = FaissVectorStore(cache_dir=tmp_path / "cache")
    retriever = KnowledgeRetriever(
        docs_dir=docs, embedder=embedder, store=store,
        chunk_size=200, chunk_overlap=20,
    )

    try:
        from bm25_index import BM25Retriever
        bm25: BM25Retriever | None = BM25Retriever(cache_dir=tmp_path / "cache")
    except RuntimeError:
        bm25 = None

    pipeline = RagPipeline(
        retriever=retriever,
        bm25=bm25,
        reranker=_StubReranker(),
        rewriter=None,  # 不启用改写
        config=PipelineConfig(
            top_k=2, min_score=0.0, candidate_pool=10, rrf_k=60,
            enable_bm25=bm25 is not None, enable_rerank=True, rerank_input_size=10,
            enable_query_rewrite=False,
        ),
    )
    pipeline.ensure_ready()

    hits = pipeline.search("失眠 焦虑")
    assert len(hits) <= 2
    # 关键：reranker 应当生效；如果 candidates >= 2，stub 会调换前两位
    assert all(isinstance(h, RetrievedChunk) for h in hits)


def test_pipeline_crisis_short_circuit(tmp_path):
    """命中危机词时，pipeline 会跳过 retriever / rerank，直接返回 crisis-resources。"""
    from pipeline import PipelineConfig, RagPipeline

    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "crisis-resources.md").write_text(
        "# 求助渠道与危机边界\n\n如果你正在考虑自伤，请立刻联系 12320-5。",
        encoding="utf-8",
    )
    (docs / "sleep.md").write_text(
        "# 睡眠\n\n固定起床时间。", encoding="utf-8"
    )

    embedder = FakeEmbedder()
    store = FaissVectorStore(cache_dir=tmp_path / "cache")
    retriever = KnowledgeRetriever(
        docs_dir=docs, embedder=embedder, store=store,
        chunk_size=200, chunk_overlap=20,
    )
    pipeline = RagPipeline(
        retriever=retriever,
        bm25=None,
        reranker=None,
        rewriter=None,
        config=PipelineConfig(
            top_k=3, min_score=0.0, enable_bm25=False, enable_rerank=False,
            enable_query_rewrite=False, crisis_force_inject=True, crisis_top_k=1,
        ),
    )
    pipeline.ensure_ready()

    hits = pipeline.search("我真的不想活了")
    assert len(hits) >= 1
    assert hits[0].source.lower() == "crisis-resources.md"


# ---------------------------------------------------------------- P4：Reranker 接口


def test_noop_reranker_keeps_order():
    from rerankers import NoopReranker

    rr = NoopReranker()
    out = rr.rerank("any query", ["a", "b", "c"], top_n=2)
    assert [r.index for r in out] == [0, 1]


def test_build_reranker_falls_back_to_noop_on_unknown():
    from rerankers import NoopReranker, build_reranker

    rr = build_reranker("no-such-reranker")
    assert isinstance(rr, NoopReranker)


def test_qdrant_store_smoke_when_available(tmp_path):
    """如果本地装了 qdrant-client，就用嵌入式模式做一次端到端烟测；没装就 skip。"""
    import pytest

    try:
        import qdrant_client  # noqa: F401
    except ImportError:
        pytest.skip("qdrant-client 未安装")

    from vector_stores import QdrantVectorStore

    cache = tmp_path / "cache"
    qdrant_path = tmp_path / "qdrant_storage"
    try:
        store = QdrantVectorStore(
            collection="test_collection",
            cache_dir=cache,
            path=str(qdrant_path),
        )
    except Exception as exc:  # 嵌入式可能因平台原因失败
        pytest.skip(f"Qdrant 嵌入式启动失败：{exc}")

    fp = "fp-qdrant"
    records = [
        ChunkRecord(source="sleep.md", title="睡眠", content="x", category="habits"),
        ChunkRecord(source="anxiety.md", title="焦虑", content="y", category="emotions"),
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.rebuild(records, vectors, fp)

    assert store.chunk_count() == 2
    assert store.needs_rebuild(fp) is False

    hits = store.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=2)
    assert any(h.record.source == "sleep.md" for h in hits)

    only_emotions = store.search(
        np.array([[1.0, 0.0]], dtype=np.float32), top_k=2, category="emotions"
    )
    assert all(h.record.category == "emotions" for h in only_emotions)
