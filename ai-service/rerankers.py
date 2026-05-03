"""
重排序模块（P4 引入）。

候选池来自 hybrid 召回（dense + sparse），但向量分数不能精确反映"query 与 doc 是否真的相关"。
重排序使用 cross-encoder / 大模型对 (query, doc) 二元组打分，把真正相关的片段顶到最前。

实现：
- ``QwenLocalReranker``：基于 Qwen3-Reranker-0.6B（causal LM 范式，按官方 README 用 yes/no 概率打分）。
- ``NoopReranker``：兜底实现，按候选原顺序返回，便于在没有模型 / 资源不足时降级。
- ``build_reranker``：工厂函数，未知 / 失败一律降级为 NoopReranker。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankResult:
    """重排序输出：原 docs 列表中的位置 + 打分（已按 score 降序）。"""

    index: int
    score: float


@runtime_checkable
class Reranker(Protocol):
    backend_name: str

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[RerankResult]:
        ...

    def describe(self) -> str:
        ...


# ---------------------------------------------------------------------------- Noop


class NoopReranker:
    """不做任何重排序，按原顺序返回 top_n。"""

    backend_name = "noop"

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[RerankResult]:
        return [RerankResult(index=i, score=float(len(docs) - i)) for i in range(min(top_n, len(docs)))]

    def describe(self) -> str:
        return "noop-reranker"


# ---------------------------------------------------------------------------- Qwen3-Reranker 本地实现


_DEFAULT_INSTRUCTION = (
    "Given a user message about mental health, retrieve passages from a Chinese mental-health "
    "self-help knowledge base that are most useful to address the user's situation."
)

_QWEN_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)
_QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class QwenLocalReranker:
    """本地千问 Qwen3-Reranker-0.6B / 4B / 8B。

    Qwen3-Reranker 是 causal LM，不是标准 cross-encoder，需要：
    1. 拼成固定 prompt（system + instruct + query + doc）。
    2. 让模型输出下一个 token，对 yes/no token 的 logit 做 softmax，取 yes 概率作为相关度。
    """

    backend_name = "qwen-local"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        *,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = "auto",
        max_length: int = 4096,
        batch_size: int = 8,
        instruction: str = _DEFAULT_INSTRUCTION,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_length = max(512, int(max_length))
        self.batch_size = max(1, int(batch_size))
        self.instruction = instruction
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._yes_id: Optional[int] = None
        self._no_id: Optional[int] = None
        self._prefix_tokens: list[int] = []
        self._suffix_tokens: list[int] = []
        self._device_resolved: Optional[str] = None

    def describe(self) -> str:
        return (
            f"qwen-local-reranker(model={self.model_name}, device={self._device_resolved or 'auto'}, "
            f"max_length={self.max_length}, batch={self.batch_size})"
        )

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[RerankResult]:
        if not docs:
            return []
        if top_n <= 0:
            return []

        self._ensure_loaded()

        scores: list[float] = []
        for start in range(0, len(docs), self.batch_size):
            batch_docs = docs[start : start + self.batch_size]
            scores.extend(self._score_batch(query, batch_docs))

        # 按分数降序，返回 top_n
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
        return [RerankResult(index=i, score=scores[i]) for i in order]

    # ----- 私有

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "transformers / torch 未安装，无法使用 QwenLocalReranker。"
                    "请执行 `pip install transformers torch`。"
                ) from exc

            logger.info(
                "正在加载本地千问 reranker：%s（约 1.2GB，首次需下载，CPU 上推理 20 个候选约 2-6 秒）",
                self.model_name,
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")

            model_kwargs: dict = {}
            if self.torch_dtype:
                # transformers 4.51+ 接收字符串 'auto' / 'float16' 等
                model_kwargs["torch_dtype"] = self.torch_dtype
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            except TypeError:
                model_kwargs.pop("torch_dtype", None)
                model = AutoModelForCausalLM.from_pretrained(self.model_name)

            device = self.device
            if not device:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                model = model.to(device)
            except Exception as exc:
                logger.warning("Reranker 转移到 %s 失败，回退到 cpu：%s", device, exc)
                device = "cpu"
                model = model.to(device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._device_resolved = device
            try:
                self._yes_id = tokenizer.convert_tokens_to_ids("yes")
                self._no_id = tokenizer.convert_tokens_to_ids("no")
            except Exception:
                self._yes_id = None
                self._no_id = None
            self._prefix_tokens = tokenizer.encode(_QWEN_PREFIX, add_special_tokens=False)
            self._suffix_tokens = tokenizer.encode(_QWEN_SUFFIX, add_special_tokens=False)

    def _format_pair(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def _score_batch(self, query: str, docs: list[str]) -> list[float]:
        import torch

        assert self._tokenizer is not None and self._model is not None

        pairs = [self._format_pair(query, d) for d in docs]
        budget = max(64, self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens))
        encoded = self._tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=budget,
        )
        for i, ids in enumerate(encoded["input_ids"]):
            encoded["input_ids"][i] = self._prefix_tokens + ids + self._suffix_tokens

        inputs = self._tokenizer.pad(
            encoded,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits[:, -1, :]
            if self._yes_id is None or self._no_id is None:
                # 兜底：直接 softmax 全表，取最大概率
                probs = torch.softmax(logits, dim=-1).max(dim=-1).values
                return probs.float().cpu().tolist()
            stacked = torch.stack(
                [logits[:, self._no_id], logits[:, self._yes_id]], dim=1
            )
            log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
            yes_probs = log_probs[:, 1].exp().float().cpu().tolist()
        return yes_probs


# ---------------------------------------------------------------------------- 工厂


def build_reranker(
    kind: str,
    *,
    qwen_model_name: str = "Qwen/Qwen3-Reranker-0.6B",
    qwen_device: Optional[str] = None,
    qwen_dtype: Optional[str] = "auto",
    qwen_max_length: int = 4096,
    qwen_batch_size: int = 8,
    qwen_instruction: Optional[str] = None,
) -> Reranker:
    """根据配置构造 Reranker；未知 / 失败统一降级为 NoopReranker。"""
    normalized = (kind or "noop").strip().lower()
    if normalized in ("qwen", "qwen-local", "qwen_local"):
        try:
            return QwenLocalReranker(
                model_name=qwen_model_name,
                device=qwen_device,
                torch_dtype=qwen_dtype,
                max_length=qwen_max_length,
                batch_size=qwen_batch_size,
                instruction=qwen_instruction or _DEFAULT_INSTRUCTION,
            )
        except Exception as exc:
            logger.warning("Qwen reranker 初始化失败，回退到 noop：%s", exc)
            return NoopReranker()
    if normalized != "noop":
        logger.warning("未知的 RAG_RERANKER=%s，已回退到 noop reranker", kind)
    return NoopReranker()
