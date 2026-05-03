import json
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, AsyncIterator, Iterator, Optional, Protocol
from uuid import uuid4

# ⚠ 必须放在所有第三方依赖（尤其是 huggingface_hub / transformers / openai 等）import 之前。
# pydantic_settings 只把 .env 里的键加载到 Settings 字段，不会注入到 os.environ；
# 但 HF_ENDPOINT / HF_HUB_OFFLINE / HF_HOME / HTTPS_PROXY 等都是其他库直接读
# os.environ 的，必须显式 load 一次才能在国内镜像、代理、离线模式等场景下生效。
from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, Header, HTTPException, status  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError, OpenAI, RateLimitError  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: E402

from bm25_index import BM25Retriever  # noqa: E402
from pipeline import PipelineConfig, RagPipeline  # noqa: E402
from query_rewriter import LLMQueryRewriter  # noqa: E402
from rag import (  # noqa: E402
    KnowledgeRetriever,
    RetrievedChunk,
    build_chat_system_prompt,
    build_embedder,
    build_vector_store,
)
from rerankers import build_reranker  # noqa: E402


def _configure_logging() -> None:
    """让自定义 logger 复用 uvicorn 的 handler，确保启动期日志能输出到控制台。"""
    uvicorn_logger = logging.getLogger("uvicorn")
    handlers = uvicorn_logger.handlers or logging.getLogger().handlers
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        if handlers:
            for handler in handlers:
                root_logger.addHandler(handler)
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )
    root_logger.setLevel(logging.INFO)
    for name in ("main", "rag", "pipeline", "vector_stores", "bm25_index", "rerankers"):
        logging.getLogger(name).setLevel(logging.INFO)
    # 第三方库 HTTP 链路日志噪音很大，会把 RAG 自己的进度日志挤没；统一压到 WARNING。
    for name in ("httpx", "httpcore", "huggingface_hub", "transformers", "urllib3", "qdrant_client"):
        logging.getLogger(name).setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    deepseek_api_key: str = Field(default="")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1")
    deepseek_model: str = Field(default="deepseek-chat")
    deepseek_timeout_seconds: float = Field(default=30.0)
    ai_service_internal_token: str = Field(default="")
    ai_service_max_output_tokens: int = Field(default=1024)

    rag_enabled: bool = Field(default=True)
    rag_top_k: int = Field(default=3, ge=1, le=10)
    rag_min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    rag_chunk_size: int = Field(default=400, ge=120, le=2000)
    rag_chunk_overlap: int = Field(default=60, ge=0, le=400)
    rag_docs_dir: str = Field(default="knowledge/docs")
    rag_cache_dir: str = Field(default="knowledge/.cache")
    rag_query_cache_size: int = Field(default=256, ge=0, le=10000)

    # ---------- Embedder 选择 ----------
    # qwen-local：本地千问 Qwen3-Embedding 系列（默认，推荐）
    # dashscope：调用阿里云 DashScope 的 OpenAI 兼容 embedding 接口（远程）
    # local：通用 sentence-transformers 加载（可指向 BGE / M3E 等）
    rag_embedder: str = Field(default="qwen-local")
    # 通用 local embedder 使用的模型名称（仅在 rag_embedder=local 时生效）
    rag_model_name: str = Field(default="BAAI/bge-small-zh-v1.5")

    # ---------- 千问本地 embedding（rag_embedder=qwen-local 时生效） ----------
    qwen_embed_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    # Qwen3-Embedding 推荐 query 端走 prompt_name="query"，doc 端不传
    qwen_embed_query_prompt_name: str = Field(default="query")
    # device: cuda / cpu / mps；空字符串走 sentence-transformers 默认（auto）
    qwen_embed_device: str = Field(default="")
    # torch_dtype: auto / float16 / bfloat16 / float32；CPU 一般用 auto 即可
    qwen_embed_dtype: str = Field(default="auto")
    qwen_embed_batch_size: int = Field(default=8, ge=1, le=128)

    # ---------- DashScope 配置 ----------
    dashscope_api_key: str = Field(default="")
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    dashscope_embed_model: str = Field(default="text-embedding-v3")
    dashscope_embed_dim: int = Field(default=1024, ge=64, le=4096)
    dashscope_timeout_seconds: float = Field(default=30.0)

    # ---------- VectorStore 选择（P2） ----------
    # qdrant：优先走 Qdrant；初始化失败会自动回退到 faiss
    # faiss：本地 FAISS + numpy 兜底
    rag_vector_store: str = Field(default="qdrant")
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")
    qdrant_collection: str = Field(default="mental_health_knowledge")
    # 嵌入式模式：设置非空时忽略 qdrant_url，使用本地文件存储
    qdrant_path: str = Field(default="")
    qdrant_prefer_grpc: bool = Field(default=False)
    qdrant_timeout_seconds: float = Field(default=30.0)

    # ---------- P3 多路召回 ----------
    rag_enable_bm25: bool = Field(default=True)
    rag_candidate_pool: int = Field(default=20, ge=1, le=200)
    rag_rrf_k: int = Field(default=60, ge=1, le=200)

    # ---------- P4 重排序 ----------
    # qwen-local：本地 Qwen3-Reranker；noop：不重排
    rag_reranker: str = Field(default="qwen-local")
    rag_enable_rerank: bool = Field(default=True)
    rag_rerank_input_size: int = Field(default=20, ge=1, le=200)
    qwen_rerank_model: str = Field(default="Qwen/Qwen3-Reranker-0.6B")
    qwen_rerank_device: str = Field(default="")
    qwen_rerank_dtype: str = Field(default="auto")
    qwen_rerank_max_length: int = Field(default=4096, ge=512, le=32768)
    qwen_rerank_batch_size: int = Field(default=8, ge=1, le=64)
    qwen_rerank_instruction: str = Field(default="")

    # ---------- P5 查询改写 ----------
    rag_enable_query_rewrite: bool = Field(default=True)
    rag_enable_multi_query: bool = Field(default=True)
    rag_enable_hyde: bool = Field(default=True)
    rag_max_rewrites: int = Field(default=3, ge=1, le=8)
    rag_hyde_min_query_chars: int = Field(default=10, ge=0, le=100)
    rag_crisis_force_inject: bool = Field(default=True)
    rag_crisis_top_k: int = Field(default=2, ge=1, le=5)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class MoodAnalysisRequest(BaseModel):
    diaryId: Optional[int] = None
    content: str = Field(min_length=1, max_length=5000)
    moodScore: Optional[int] = Field(default=None, ge=1, le=10)


class MoodAnalysisResponse(BaseModel):
    analysisText: str
    model: str
    requestId: str


class ChatMessage(BaseModel):
    role: str = Field(min_length=1)
    content: str = Field(min_length=1)


class ChatStreamRequest(BaseModel):
    sessionId: Optional[int] = None
    messages: list[ChatMessage] = Field(min_length=1)


class ChatStreamEvent(BaseModel):
    type: str
    content: str = ""
    requestId: str = ""
    errorCode: str = ""
    message: str = ""
    retryable: Optional[bool] = None


class HealthResponse(BaseModel):
    status: str
    model: str


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    topK: Optional[int] = Field(default=None, ge=1, le=10)
    minScore: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class RagSearchHit(BaseModel):
    source: str
    title: str
    content: str
    score: float


class RagSearchResponse(BaseModel):
    hits: list[RagSearchHit]


class RagReindexResponse(BaseModel):
    chunkCount: int
    model: str


class AIServiceError(Exception):
    def __init__(self, status_code: int, error_code: str, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.retryable = retryable


class AIServiceClient(Protocol):
    def analyze_diary(self, payload: MoodAnalysisRequest) -> MoodAnalysisResponse:
        ...

    def start_chat_stream(self, payload: ChatStreamRequest) -> Iterator[ChatStreamEvent]:
        ...


class DeepSeekAIService:
    def __init__(
        self,
        settings: Settings,
        retriever: Optional["RagPipeline"] = None,
    ) -> None:
        self.settings = settings
        # 名字仍叫 retriever 是为了保持单测的依赖覆盖兼容；类型其实已是 RagPipeline。
        self.retriever = retriever
        self.client = OpenAI(
            api_key=settings.deepseek_api_key or "missing-key",
            base_url=settings.deepseek_base_url,
            timeout=settings.deepseek_timeout_seconds,
        )

    def analyze_diary(self, payload: MoodAnalysisRequest) -> MoodAnalysisResponse:
        self._ensure_api_key()
        request_id = str(uuid4())
        user_message = self._build_mood_prompt(payload)

        try:
            completion = self.client.chat.completions.create(
                model=self.settings.deepseek_model,
                temperature=0.7,
                max_tokens=self.settings.ai_service_max_output_tokens,
                messages=[
                    {"role": "system", "content": self._mood_system_prompt()},
                    {"role": "user", "content": user_message},
                ],
            )
        except Exception as exc:
            raise self._map_exception(exc) from exc

        content = completion.choices[0].message.content if completion.choices else None
        if not content or not content.strip():
            raise AIServiceError(502, "EMPTY_RESPONSE", "DeepSeek 返回了空内容", True)

        return MoodAnalysisResponse(
            analysisText=content.strip(),
            model=completion.model or self.settings.deepseek_model,
            requestId=completion.id or request_id,
        )

    def start_chat_stream(self, payload: ChatStreamRequest) -> Iterator[ChatStreamEvent]:
        self._ensure_api_key()

        system_prompt = self._build_chat_system_prompt_with_rag(payload)

        try:
            upstream_stream = self.client.chat.completions.create(
                model=self.settings.deepseek_model,
                temperature=0.7,
                max_tokens=self.settings.ai_service_max_output_tokens,
                stream=True,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *[
                        {"role": message.role, "content": message.content.strip()}
                        for message in payload.messages
                    ],
                ],
            )
        except Exception as exc:
            raise self._map_exception(exc) from exc

        def iterator() -> Iterator[ChatStreamEvent]:
            request_id = str(uuid4())
            has_delta = False

            try:
                for chunk in upstream_stream:
                    if getattr(chunk, "id", None):
                        request_id = chunk.id

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta.content if choice.delta else None
                    if delta:
                        has_delta = True
                        yield ChatStreamEvent(
                            type="delta",
                            content=delta,
                            requestId=request_id,
                        )

                if not has_delta:
                    yield ChatStreamEvent(
                        type="error",
                        requestId=request_id,
                        errorCode="EMPTY_RESPONSE",
                        message="DeepSeek 未返回有效内容",
                        retryable=True,
                    )
                    return

                yield ChatStreamEvent(type="done", requestId=request_id)
            except GeneratorExit:
                return
            except Exception as exc:
                mapped = self._map_exception(exc)
                yield ChatStreamEvent(
                    type="error",
                    requestId=request_id,
                    errorCode=mapped.error_code,
                    message=mapped.message,
                    retryable=mapped.retryable,
                )

        return iterator()

    def _build_chat_system_prompt_with_rag(self, payload: ChatStreamRequest) -> str:
        base_prompt = self._chat_system_prompt()
        if not self.settings.rag_enabled or self.retriever is None:
            return base_prompt

        last_user_message = next(
            (m for m in reversed(payload.messages) if (m.role or "").lower() == "user"),
            None,
        )
        if last_user_message is None:
            return base_prompt

        history = [m for m in payload.messages if m is not last_user_message]
        try:
            hits = self.retriever.search(
                last_user_message.content,
                history=history,
                top_k=self.settings.rag_top_k,
                min_score=self.settings.rag_min_score,
            )
        except Exception as exc:  # 检索失败不阻塞对话
            logger.warning("RAG 检索失败，降级为无知识增强：%s", exc)
            return base_prompt

        if not hits:
            return base_prompt
        return build_chat_system_prompt(base_prompt, hits)

    def _ensure_api_key(self) -> None:
        if not self.settings.deepseek_api_key:
            raise AIServiceError(503, "MISSING_API_KEY", "DeepSeek API Key 未配置", False)

    @staticmethod
    def _mood_system_prompt() -> str:
        return (
            "你是一名温和、谨慎的心理健康陪伴助手。"
            "请基于用户的日记内容做简短情绪分析，语气需温柔、尊重、非评判。"
            "不要做医学诊断，不要给出绝对化结论，不要夸大风险。"
            "输出 2 到 4 句话，先共情总结，再给出 1 到 2 条具体、可执行的自我关怀建议。"
            "如果用户情绪明显低落，只能建议寻求可信赖的人或专业支持，不能替代专业诊疗。"
        )

    @staticmethod
    def _chat_system_prompt() -> str:
        return (
            "你是一名温和、谨慎、耐心、有专业心理学背景的心理健康陪伴助手。"
            "请使用中文进行对话，优先共情和倾听，再帮助用户理解自己的状态，"
            "最后给出具体、可操作的方法。"
            "不要做医学诊断，不要承诺治愈，不要给出危险或极端建议。"
            "回答风格："
            "（1）至少 3~5 个自然段，避免一两句话就结束；"
            "（2）共情 → 澄清/解释 → 2~4 条具体可执行的方法或练习 → 鼓励的收尾；"
            "（3）方法部分要真正展开，告诉用户每一步该怎么做、为什么这样做，"
            "避免只丢方法名（例如不要只说『试试正念呼吸』，而要写出具体步骤）；"
            "（4）整体语气保持温柔、尊重、非评判，自然分段，适合逐步流式输出。"
        )

    @staticmethod
    def _build_mood_prompt(payload: MoodAnalysisRequest) -> str:
        mood_part = "未提供" if payload.moodScore is None else str(payload.moodScore)
        return (
            f"用户日记内容：\n{payload.content.strip()}\n\n"
            f"用户自评情绪分值（1-10）：{mood_part}\n"
            "请输出一段适合直接展示给用户的中文分析文案。"
        )

    @staticmethod
    def _map_exception(exc: Exception) -> AIServiceError:
        if isinstance(exc, APITimeoutError):
            return AIServiceError(504, "DEEPSEEK_TIMEOUT", "DeepSeek 调用超时", True)
        if isinstance(exc, APIConnectionError):
            return AIServiceError(503, "DEEPSEEK_UNREACHABLE", "无法连接 DeepSeek 服务", True)
        if isinstance(exc, RateLimitError):
            return AIServiceError(429, "DEEPSEEK_RATE_LIMIT", "DeepSeek 调用频率过高", True)
        if isinstance(exc, BadRequestError):
            return AIServiceError(400, "DEEPSEEK_BAD_REQUEST", "提交给 DeepSeek 的请求无效", False)
        if isinstance(exc, APIStatusError):
            return AIServiceError(502, "DEEPSEEK_UPSTREAM_ERROR", "DeepSeek 返回异常状态", True)
        if isinstance(exc, AIServiceError):
            return exc
        return AIServiceError(502, "DEEPSEEK_UNKNOWN_ERROR", "DeepSeek 调用失败", True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


_PIPELINE_CACHE: dict[str, Optional[RagPipeline]] = {}


def _pipeline_cache_key(settings: Settings) -> str:
    # 把所有影响 pipeline 构造的字段拼成一个 key；任意一项变化就重建
    return "|".join(
        [
            settings.rag_docs_dir,
            settings.rag_cache_dir,
            str(settings.rag_chunk_size),
            str(settings.rag_chunk_overlap),
            str(settings.rag_query_cache_size),
            settings.rag_embedder,
            settings.rag_model_name,
            settings.dashscope_api_key,
            settings.dashscope_embed_model,
            str(settings.dashscope_embed_dim),
            settings.dashscope_base_url,
            str(settings.dashscope_timeout_seconds),
            settings.qwen_embed_model,
            settings.qwen_embed_query_prompt_name,
            settings.qwen_embed_device,
            settings.qwen_embed_dtype,
            str(settings.qwen_embed_batch_size),
            settings.rag_vector_store,
            settings.qdrant_url,
            settings.qdrant_api_key,
            settings.qdrant_collection,
            settings.qdrant_path,
            str(settings.qdrant_prefer_grpc),
            str(settings.qdrant_timeout_seconds),
            str(settings.rag_enable_bm25),
            str(settings.rag_candidate_pool),
            str(settings.rag_rrf_k),
            settings.rag_reranker,
            str(settings.rag_enable_rerank),
            str(settings.rag_rerank_input_size),
            settings.qwen_rerank_model,
            settings.qwen_rerank_device,
            settings.qwen_rerank_dtype,
            str(settings.qwen_rerank_max_length),
            str(settings.qwen_rerank_batch_size),
            settings.qwen_rerank_instruction,
            str(settings.rag_enable_query_rewrite),
            str(settings.rag_enable_multi_query),
            str(settings.rag_enable_hyde),
            str(settings.rag_max_rewrites),
            str(settings.rag_hyde_min_query_chars),
            str(settings.rag_crisis_force_inject),
            str(settings.rag_crisis_top_k),
            str(settings.rag_top_k),
            str(settings.rag_min_score),
        ]
    )


def _build_pipeline(settings: Settings) -> Optional[RagPipeline]:
    """构造完整 RAG pipeline；任意核心组件失败都返回 None 以便上层降级。"""
    try:
        embedder = build_embedder(
            settings.rag_embedder,
            local_model_name=settings.rag_model_name,
            dashscope_api_key=settings.dashscope_api_key,
            dashscope_model=settings.dashscope_embed_model,
            dashscope_dim=settings.dashscope_embed_dim,
            dashscope_base_url=settings.dashscope_base_url,
            dashscope_timeout=settings.dashscope_timeout_seconds,
            qwen_model_name=settings.qwen_embed_model,
            qwen_query_prompt_name=settings.qwen_embed_query_prompt_name or None,
            qwen_device=settings.qwen_embed_device or None,
            qwen_torch_dtype=settings.qwen_embed_dtype or None,
            qwen_batch_size=settings.qwen_embed_batch_size,
        )
        store = build_vector_store(
            settings.rag_vector_store,
            cache_dir=Path(settings.rag_cache_dir),
            qdrant_collection=settings.qdrant_collection,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            qdrant_path=settings.qdrant_path,
            qdrant_prefer_grpc=settings.qdrant_prefer_grpc,
            qdrant_timeout=settings.qdrant_timeout_seconds,
        )
        retriever = KnowledgeRetriever(
            docs_dir=Path(settings.rag_docs_dir),
            embedder=embedder,
            store=store,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            query_cache_size=settings.rag_query_cache_size,
        )

        bm25: Optional[BM25Retriever] = None
        if settings.rag_enable_bm25:
            try:
                bm25 = BM25Retriever(cache_dir=Path(settings.rag_cache_dir))
            except Exception as exc:
                logger.warning("BM25 初始化失败，跳过稀疏召回：%s", exc)
                bm25 = None

        reranker = build_reranker(
            settings.rag_reranker if settings.rag_enable_rerank else "noop",
            qwen_model_name=settings.qwen_rerank_model,
            qwen_device=settings.qwen_rerank_device or None,
            qwen_dtype=settings.qwen_rerank_dtype or None,
            qwen_max_length=settings.qwen_rerank_max_length,
            qwen_batch_size=settings.qwen_rerank_batch_size,
            qwen_instruction=settings.qwen_rerank_instruction or None,
        )

        rewriter: Optional[LLMQueryRewriter] = None
        if settings.rag_enable_query_rewrite and settings.deepseek_api_key:
            try:
                from openai import OpenAI

                rewrite_client = OpenAI(
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                    timeout=settings.deepseek_timeout_seconds,
                )
                rewriter = LLMQueryRewriter(
                    client=rewrite_client,
                    model=settings.deepseek_model,
                    enable_multi_query=settings.rag_enable_multi_query,
                    enable_hyde=settings.rag_enable_hyde,
                    max_queries=settings.rag_max_rewrites,
                    hyde_min_query_chars=settings.rag_hyde_min_query_chars,
                )
            except Exception as exc:
                logger.warning("Query rewriter 初始化失败，跳过改写：%s", exc)
                rewriter = None

        config = PipelineConfig(
            top_k=settings.rag_top_k,
            min_score=settings.rag_min_score,
            candidate_pool=settings.rag_candidate_pool,
            rrf_k=settings.rag_rrf_k,
            enable_bm25=settings.rag_enable_bm25 and bm25 is not None,
            enable_rerank=settings.rag_enable_rerank,
            rerank_input_size=settings.rag_rerank_input_size,
            enable_query_rewrite=settings.rag_enable_query_rewrite and rewriter is not None,
            crisis_top_k=settings.rag_crisis_top_k,
            crisis_force_inject=settings.rag_crisis_force_inject,
        )

        pipeline = RagPipeline(
            retriever=retriever,
            bm25=bm25,
            reranker=reranker,
            rewriter=rewriter,
            config=config,
        )
        count = pipeline.ensure_ready()
        logger.info(
            "RAG pipeline 初始化完成：embedder=%s，store=%s，bm25=%s，reranker=%s，rewriter=%s，知识片段数=%d",
            embedder.identifier,
            store.describe(),
            "on" if bm25 is not None else "off",
            reranker.describe(),
            "on" if rewriter is not None else "off",
            count,
        )
        return pipeline
    except Exception as exc:
        logger.warning("RAG pipeline 初始化失败，将降级为无知识增强：%s", exc)
        return None


def _pipeline_from_settings(settings: Settings) -> Optional[RagPipeline]:
    key = _pipeline_cache_key(settings)
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = _build_pipeline(settings)
    return _PIPELINE_CACHE[key]


# 旧名字保留，作为 _pipeline_from_settings 的别名（避免破坏其他模块的引用）
def _retriever_from_settings(settings: Settings) -> Optional[RagPipeline]:
    return _pipeline_from_settings(settings)


def get_retriever(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Optional[RagPipeline]:
    """名字保留为 retriever 是为了让 FastAPI 依赖覆盖与单测无须改动。

    实际返回的是 ``RagPipeline``，对外仍提供 ``search(query, top_k, min_score, ...)`` 与
    ``ensure_ready / rebuild`` 方法，对调用方无感知。
    """
    if not settings.rag_enabled:
        return None
    return _pipeline_from_settings(settings)


def get_ai_service(
    settings: Annotated[Settings, Depends(get_settings)],
    retriever: Annotated[Optional[RagPipeline], Depends(get_retriever)],
) -> AIServiceClient:
    return DeepSeekAIService(settings, retriever)


def verify_internal_token(
    settings: Annotated[Settings, Depends(get_settings)],
    x_internal_token: Annotated[Optional[str], Header(alias="X-Internal-Token")] = None,
) -> None:
    expected = settings.ai_service_internal_token.strip()
    if expected and x_internal_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "errorCode": "UNAUTHORIZED",
                "message": "未授权访问 AI 服务",
                "retryable": False,
            },
        )


def encode_sse(event: ChatStreamEvent) -> bytes:
    payload = json.dumps(event.model_dump(exclude_none=True), ensure_ascii=False)
    return f"data: {payload}\n\n".encode("utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _configure_logging()
    settings = get_settings()
    if settings.rag_enabled:
        kind = settings.rag_embedder.lower()
        if kind == "dashscope":
            embedder_label = f"dashscope::{settings.dashscope_embed_model}"
        elif kind in ("qwen", "qwen-local", "qwen_local"):
            embedder_label = f"qwen-local::{settings.qwen_embed_model}"
        else:
            embedder_label = f"local::{settings.rag_model_name}"
        store_label = settings.rag_vector_store.lower()
        if store_label == "qdrant":
            store_label = (
                f"qdrant(path={settings.qdrant_path})"
                if settings.qdrant_path
                else f"qdrant({settings.qdrant_url}/{settings.qdrant_collection})"
            )
        logger.info(
            "RAG 启动预热：docs_dir=%s, embedder=%s, store=%s, chunk_size=%d, chunk_overlap=%d, top_k=%d, min_score=%.2f, query_cache=%d",
            settings.rag_docs_dir,
            embedder_label,
            store_label,
            settings.rag_chunk_size,
            settings.rag_chunk_overlap,
            settings.rag_top_k,
            settings.rag_min_score,
            settings.rag_query_cache_size,
        )
        retriever = _retriever_from_settings(settings)
        if retriever is None:
            logger.warning("RAG 预热失败，已降级为无知识增强模式")
    else:
        logger.info("RAG 未启用（RAG_ENABLED=false），跳过预热")
    yield


app = FastAPI(title="mental-health-ai-service", version="0.2.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(status="ok", model=settings.deepseek_model)


@app.post(
    "/internal/v1/mood/analyze",
    response_model=MoodAnalysisResponse,
    dependencies=[Depends(verify_internal_token)],
)
def analyze_mood(
    payload: MoodAnalysisRequest,
    ai_service: Annotated[AIServiceClient, Depends(get_ai_service)],
) -> MoodAnalysisResponse:
    try:
        return ai_service.analyze_diary(payload)
    except AIServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "errorCode": exc.error_code,
                "message": exc.message,
                "retryable": exc.retryable,
            },
        ) from exc


@app.post(
    "/internal/v1/chat/stream",
    dependencies=[Depends(verify_internal_token)],
)
def stream_chat(
    payload: ChatStreamRequest,
    ai_service: Annotated[AIServiceClient, Depends(get_ai_service)],
) -> StreamingResponse:
    try:
        event_stream = ai_service.start_chat_stream(payload)
    except AIServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "errorCode": exc.error_code,
                "message": exc.message,
                "retryable": exc.retryable,
            },
        ) from exc

    return StreamingResponse(
        (encode_sse(event) for event in event_stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post(
    "/internal/v1/rag/search",
    response_model=RagSearchResponse,
    dependencies=[Depends(verify_internal_token)],
)
def rag_search(
    payload: RagSearchRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    retriever: Annotated[Optional[RagPipeline], Depends(get_retriever)],
) -> RagSearchResponse:
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail={
                "errorCode": "RAG_DISABLED",
                "message": "RAG 未启用或初始化失败",
                "retryable": False,
            },
        )
    top_k = payload.topK or settings.rag_top_k
    min_score = payload.minScore if payload.minScore is not None else settings.rag_min_score
    try:
        hits: list[RetrievedChunk] = retriever.search(
            payload.query, top_k=top_k, min_score=min_score
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "errorCode": "RAG_SEARCH_FAILED",
                "message": f"RAG 检索失败：{exc}",
                "retryable": True,
            },
        ) from exc
    return RagSearchResponse(
        hits=[
            RagSearchHit(source=h.source, title=h.title, content=h.content, score=h.score)
            for h in hits
        ]
    )


@app.post(
    "/internal/v1/rag/reindex",
    response_model=RagReindexResponse,
    dependencies=[Depends(verify_internal_token)],
)
def rag_reindex(
    settings: Annotated[Settings, Depends(get_settings)],
    retriever: Annotated[Optional[RagPipeline], Depends(get_retriever)],
) -> RagReindexResponse:
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail={
                "errorCode": "RAG_DISABLED",
                "message": "RAG 未启用或初始化失败",
                "retryable": False,
            },
        )
    try:
        # pipeline 暴露 rebuild()，单测里的 FakeRetriever 仍然提供 ensure_index(force_rebuild=True)
        rebuild = getattr(retriever, "rebuild", None)
        if callable(rebuild):
            count = rebuild()
        else:
            count = retriever.ensure_index(force_rebuild=True)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "errorCode": "RAG_REINDEX_FAILED",
                "message": f"RAG 重建索引失败：{exc}",
                "retryable": True,
            },
        ) from exc
    return RagReindexResponse(chunkCount=count, model=settings.rag_model_name)
