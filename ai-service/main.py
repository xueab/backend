import json
from functools import lru_cache
from typing import Annotated, Iterator, Optional, Protocol
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    deepseek_api_key: str = Field(default="")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1")
    deepseek_model: str = Field(default="deepseek-chat")
    deepseek_timeout_seconds: float = Field(default=30.0)
    ai_service_internal_token: str = Field(default="")
    ai_service_max_output_tokens: int = Field(default=320)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
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
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
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

        try:
            upstream_stream = self.client.chat.completions.create(
                model=self.settings.deepseek_model,
                temperature=0.7,
                max_tokens=self.settings.ai_service_max_output_tokens,
                stream=True,
                messages=[
                    {"role": "system", "content": self._chat_system_prompt()},
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
            "你是一名温和、谨慎、耐心的心理健康陪伴助手。"
            "请使用中文进行对话，优先共情、澄清和支持。"
            "不要做医学诊断，不要承诺治愈，不要给出危险或极端建议。"
            "在用户表达压力、焦虑、低落时，请给出简洁、可执行、低风险的陪伴式建议。"
            "回答尽量自然分段，适合逐步流式输出。"
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


def get_ai_service(settings: Annotated[Settings, Depends(get_settings)]) -> AIServiceClient:
    return DeepSeekAIService(settings)


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


app = FastAPI(title="mental-health-ai-service", version="0.1.0")


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
