"""
查询改写与安全兜底（P5 引入）。

针对中文聊天的"模糊输入"问题：
1. **危机词检测**（``detect_crisis``）：识别自杀 / 自残 / 极端关键词，由调用方决定是否
   绕过相似度检索强制注入 ``crisis-resources``。
2. **Multi-query 改写**（``rewrite_multi_query``）：用 LLM 把口语化、含糊的输入改写成
   2~3 条更适合知识库检索的查询；保留原查询作为第一条。
3. **HyDE**（``hyde_expand``）：让 LLM 先写一段假想的回答，用回答的语义去检索（对超短、
   无主题词的输入特别有效）。

为了避免阻塞主流程，所有 LLM 调用都做了：
- 严格超时与异常隔离（任一步失败都退化为只用原始查询）。
- 输出长度上限（避免 prompt 越改越长）。
- 输入长度截断（用户消息 / 历史只取最近若干字符）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- 危机词


# 中文危机词；命中任一即认为需要走安全兜底。
# 这里不追求覆盖所有场景，仅做"高召回的硬触发"，最终安全话术由 system prompt + crisis-resources 知识保证。
_CRISIS_PATTERNS = [
    re.compile(p)
    for p in (
        r"自杀",
        r"想死",
        r"不想活",
        r"活不下去",
        r"结束生命",
        r"了断",
        r"自残",
        r"割腕",
        r"跳楼",
        r"安眠药.{0,4}(吃|过量|攒)",
        r"跳河",
        r"上吊",
        r"枪.{0,3}(自己)",
        r"伤害自己",
    )
]


def detect_crisis(query: str) -> bool:
    """命中任一危机模式返回 True。空字符串 / None 返回 False。"""
    if not query:
        return False
    text = query.strip()
    if not text:
        return False
    return any(p.search(text) for p in _CRISIS_PATTERNS)


# ---------------------------------------------------------------------------- 改写结果数据结构


@dataclass
class EnrichedQuery:
    """改写后的查询包，供下游 retriever 消费。

    - ``raw``：原始用户输入。
    - ``queries``：用于多路召回的查询列表，第一条恒为 raw 或其精简版。
    - ``hyde``：可选的假设答案（用于 HyDE 检索；为空表示未启用）。
    - ``is_crisis``：是否触发危机兜底。
    """

    raw: str
    queries: list[str] = field(default_factory=list)
    hyde: Optional[str] = None
    is_crisis: bool = False


# ---------------------------------------------------------------------------- LLM 改写器


class LLMQueryRewriter:
    """基于现有 OpenAI 兼容客户端（DeepSeek / DashScope / OpenAI 都可）的查询改写器。"""

    def __init__(
        self,
        client,
        model: str,
        *,
        enable_multi_query: bool = True,
        enable_hyde: bool = True,
        max_queries: int = 3,
        history_chars: int = 600,
        query_chars: int = 500,
        rewrite_max_tokens: int = 200,
        hyde_max_tokens: int = 160,
        hyde_min_query_chars: int = 8,
        hyde_temperature: float = 0.3,
        rewrite_temperature: float = 0.2,
    ) -> None:
        self.client = client
        self.model = model
        self.enable_multi_query = enable_multi_query
        self.enable_hyde = enable_hyde
        self.max_queries = max(1, int(max_queries))
        self.history_chars = max(0, int(history_chars))
        self.query_chars = max(50, int(query_chars))
        self.rewrite_max_tokens = max(50, int(rewrite_max_tokens))
        self.hyde_max_tokens = max(50, int(hyde_max_tokens))
        self.hyde_min_query_chars = max(0, int(hyde_min_query_chars))
        self.hyde_temperature = float(hyde_temperature)
        self.rewrite_temperature = float(rewrite_temperature)

    def enrich(
        self,
        raw_query: str,
        history: Optional[list] = None,
    ) -> EnrichedQuery:
        """组合：危机检测 + multi-query 改写 + HyDE。任意子步骤失败都不影响主流程。"""
        text = (raw_query or "").strip()
        result = EnrichedQuery(raw=text, queries=[text] if text else [])

        if not text:
            return result

        result.is_crisis = detect_crisis(text)
        if result.is_crisis:
            # 危机场景下不再做改写 / HyDE，避免引入噪音；交给 pipeline 强制兜底
            return result

        history_brief = self._format_history(history)

        if self.enable_multi_query:
            try:
                rewrites = self._multi_query(text, history_brief)
            except Exception as exc:
                logger.warning("Multi-query 改写失败，仅使用原始查询：%s", exc)
                rewrites = []
            for q in rewrites:
                if q and q not in result.queries and len(result.queries) < self.max_queries:
                    result.queries.append(q)

        if self.enable_hyde and len(text) <= max(self.hyde_min_query_chars, 8):
            try:
                result.hyde = self._hyde(text, history_brief)
            except Exception as exc:
                logger.warning("HyDE 生成失败，跳过：%s", exc)
                result.hyde = None

        return result

    # ----- 私有

    def _format_history(self, history) -> str:
        if not history:
            return ""
        # history 可能是 ChatMessage 列表也可能是 dict 列表，做一次鸭子类型适配
        recent = history[-6:]
        parts: list[str] = []
        for m in recent:
            role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
            if not content:
                continue
            content = str(content).strip().replace("\n", " ")
            if not content:
                continue
            parts.append(f"{role or 'user'}: {content[:200]}")
        joined = "\n".join(parts)
        if len(joined) > self.history_chars:
            joined = "…" + joined[-self.history_chars :]
        return joined

    def _multi_query(self, raw_query: str, history_brief: str) -> list[str]:
        prompt = self._build_rewrite_prompt(raw_query[: self.query_chars], history_brief)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.rewrite_temperature,
            max_tokens=self.rewrite_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content if resp.choices else ""
        if not content:
            return []
        return self._parse_rewrites(content)

    def _hyde(self, raw_query: str, history_brief: str) -> Optional[str]:
        prompt = self._build_hyde_prompt(raw_query[: self.query_chars], history_brief)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.hyde_temperature,
            max_tokens=self.hyde_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content if resp.choices else ""
        content = (content or "").strip()
        return content or None

    @staticmethod
    def _build_rewrite_prompt(raw_query: str, history_brief: str) -> str:
        history_block = (
            f"以下是最近几轮聊天历史（仅供参考，不要复述）：\n{history_brief}\n\n"
            if history_brief
            else ""
        )
        return (
            "你是一名中文心理健康知识库的检索助手。请把用户的最新一条消息改写成 2~3 条"
            "更适合在心理健康知识库中做向量 / 关键词检索的中文短语。\n"
            "要求：\n"
            "1. 每条单独一行，不要编号、不要解释、不要标点开头。\n"
            "2. 第 1 条尽量贴近用户原意。\n"
            "3. 第 2~3 条可以补充更明确的主题词（焦虑 / 失眠 / 关系 / 自我评价 等）。\n"
            "4. 不要出现单纯的情绪化语气词（如\"啊\"、\"唉\"、\"呜呜\" 等）。\n"
            "5. 每条不要超过 30 个汉字。\n\n"
            f"{history_block}用户最新消息：{raw_query}\n\n"
            "改写后的查询（每行一条）："
        )

    @staticmethod
    def _build_hyde_prompt(raw_query: str, history_brief: str) -> str:
        history_block = (
            f"最近的对话片段：\n{history_brief}\n\n" if history_brief else ""
        )
        return (
            "你是一名温和、谨慎的中文心理健康陪伴助手。"
            "请用 80 字以内为下面的用户消息写一段你会给出的回答（用于检索向量构造，"
            "不要免责声明，不要诊断，不要罗列编号）。\n\n"
            f"{history_block}用户消息：{raw_query}\n\n回答："
        )

    @staticmethod
    def _parse_rewrites(content: str) -> list[str]:
        results: list[str] = []
        for line in content.splitlines():
            cleaned = line.strip()
            cleaned = re.sub(r"^[\-\*\u2022\u25E6\u25CF\d\.\、\s\)）]+", "", cleaned)
            cleaned = cleaned.strip().strip("\"'`“”‘’")
            if not cleaned or len(cleaned) > 60:
                continue
            results.append(cleaned)
            if len(results) >= 5:
                break
        return results
