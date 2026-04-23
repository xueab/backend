from fastapi.testclient import TestClient

from main import (
    ChatStreamEvent,
    ChatStreamRequest,
    MoodAnalysisRequest,
    MoodAnalysisResponse,
    app,
    get_ai_service,
    get_settings,
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


def override_settings():
    class TestSettings:
        deepseek_model = "deepseek-chat"
        ai_service_internal_token = ""

    return TestSettings()


def setup_client() -> TestClient:
    app.dependency_overrides[get_settings] = override_settings
    app.dependency_overrides[get_ai_service] = lambda: FakeAIService()
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
