package com.mentalhealth.backend.service.impl;

import com.mentalhealth.backend.common.BizException;
import com.mentalhealth.backend.config.AiServiceProperties;
import com.mentalhealth.backend.dto.AiDiaryAnalysisRequest;
import com.mentalhealth.backend.dto.AiDiaryAnalysisResponse;
import com.mentalhealth.backend.entity.MoodDiary;
import com.mentalhealth.backend.service.AiAnalysisService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;

@Service
@RequiredArgsConstructor
public class RemoteAiAnalysisService implements AiAnalysisService {

    private final RestClient aiServiceRestClient;
    private final AiServiceProperties aiServiceProperties;

    @Override
    public String analyze(MoodDiary diary) {
        if (diary == null || !StringUtils.hasText(diary.getContent())) {
            throw new BizException("日记内容为空，无法进行 AI 分析");
        }

        AiDiaryAnalysisRequest request = new AiDiaryAnalysisRequest(
                diary.getId(),
                diary.getContent().trim(),
                diary.getMoodScore()
        );

        try {
            AiDiaryAnalysisResponse response = aiServiceRestClient.post()
                    .uri(aiServiceProperties.getAnalyzePath())
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(request)
                    .retrieve()
                    .body(AiDiaryAnalysisResponse.class);

            if (response == null || !StringUtils.hasText(response.analysisText())) {
                throw new BizException(502, "AI 服务返回了空内容");
            }

            return response.analysisText().trim();
        } catch (ResourceAccessException e) {
            throw new BizException(503, "AI 服务连接超时，请稍后重试");
        } catch (RestClientResponseException e) {
            throw new BizException(502, "AI 服务调用失败，请稍后重试");
        } catch (RestClientException e) {
            throw new BizException(502, "AI 服务暂不可用，请稍后重试");
        }
    }
}
