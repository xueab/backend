package com.mentalhealth.backend.service;

import com.mentalhealth.backend.common.BizException;
import com.mentalhealth.backend.config.AiServiceProperties;
import com.mentalhealth.backend.entity.MoodDiary;
import com.mentalhealth.backend.service.impl.RemoteAiAnalysisService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withServerError;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

class RemoteAiAnalysisServiceTest {

    private MockRestServiceServer server;
    private RemoteAiAnalysisService service;

    @BeforeEach
    void setUp() {
        RestClient.Builder builder = RestClient.builder().baseUrl("http://localhost:8001");
        server = MockRestServiceServer.bindTo(builder).build();

        AiServiceProperties properties = new AiServiceProperties();
        properties.setAnalyzePath("/internal/v1/mood/analyze");

        service = new RemoteAiAnalysisService(builder.build(), properties);
    }

    @Test
    void analyze_success_returnsAnalysisText() {
        server.expect(requestTo("http://localhost:8001/internal/v1/mood/analyze"))
                .andExpect(method(HttpMethod.POST))
                .andRespond(withSuccess("""
                        {
                          "analysisText": "请先肯定自己今天的努力，再安排一次短暂休息。",
                          "model": "deepseek-chat",
                          "requestId": "req-1"
                        }
                        """, MediaType.APPLICATION_JSON));

        String result = service.analyze(sampleDiary());

        assertEquals("请先肯定自己今天的努力，再安排一次短暂休息。", result);
        server.verify();
    }

    @Test
    void analyze_whenAiServiceReturnsServerError_throwsFriendlyBizException() {
        server.expect(requestTo("http://localhost:8001/internal/v1/mood/analyze"))
                .andExpect(method(HttpMethod.POST))
                .andRespond(withServerError());

        BizException ex = assertThrows(BizException.class, () -> service.analyze(sampleDiary()));

        assertEquals(502, ex.getCode());
        assertEquals("AI 服务调用失败，请稍后重试", ex.getMessage());
        server.verify();
    }

    private MoodDiary sampleDiary() {
        MoodDiary diary = new MoodDiary();
        diary.setId(1L);
        diary.setContent("今天心情不太好，但还是想认真记录下来。");
        diary.setMoodScore(4);
        return diary;
    }
}
