package com.mentalhealth.backend.service.ai;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.mentalhealth.backend.common.BizException;
import com.mentalhealth.backend.config.AiServiceProperties;
import com.mentalhealth.backend.dto.AiChatStreamEvent;
import com.mentalhealth.backend.dto.AiChatStreamRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.function.Consumer;

@Service
@RequiredArgsConstructor
public class HttpAiChatStreamClient implements AiChatStreamClient {

    private final HttpClient aiStreamHttpClient;
    private final ObjectMapper objectMapper;
    private final AiServiceProperties aiServiceProperties;

    @Override
    public void streamChat(AiChatStreamRequest request, Consumer<AiChatStreamEvent> eventConsumer) {
        try {
            HttpRequest httpRequest = buildRequest(request);
            HttpResponse<InputStream> response = aiStreamHttpClient.send(
                    httpRequest,
                    HttpResponse.BodyHandlers.ofInputStream()
            );

            if (response.statusCode() >= 400) {
                throw new BizException(
                        response.statusCode(),
                        resolveUpstreamErrorMessage(response.statusCode(), response.body())
                );
            }

            consumeSseStream(response.body(), eventConsumer);
        } catch (HttpTimeoutException e) {
            throw new BizException(503, "AI 对话服务响应超时，请稍后重试");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new BizException(500, "AI 对话请求被中断");
        } catch (IOException e) {
            throw new BizException(502, "AI 对话服务暂不可用，请稍后重试");
        }
    }

    private HttpRequest buildRequest(AiChatStreamRequest request) throws JsonProcessingException {
        String json = objectMapper.writeValueAsString(request);
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(aiServiceProperties.getBaseUrl() + aiServiceProperties.getChatStreamPath()))
                .version(HttpClient.Version.HTTP_1_1)
                .timeout(Duration.ofMillis(aiServiceProperties.getStreamReadTimeoutMs()))
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8));

        if (StringUtils.hasText(aiServiceProperties.getInternalToken())) {
            builder.header("X-Internal-Token", aiServiceProperties.getInternalToken());
        }
        return builder.build();
    }

    private void consumeSseStream(InputStream responseBody, Consumer<AiChatStreamEvent> eventConsumer) throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(responseBody, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.startsWith("data:")) {
                    continue;
                }
                String payload = line.substring(5).trim();
                if (payload.isEmpty()) {
                    continue;
                }
                AiChatStreamEvent event = objectMapper.readValue(payload, AiChatStreamEvent.class);
                eventConsumer.accept(event);
            }
        }
    }

    private String resolveUpstreamErrorMessage(int statusCode, InputStream responseBody) {
        try {
            String raw = new String(responseBody.readAllBytes(), StandardCharsets.UTF_8).trim();
            if (raw.isEmpty()) {
                return fallbackErrorMessage(statusCode);
            }

            JsonNode root = objectMapper.readTree(raw);
            JsonNode detail = root.path("detail");
            if (detail.isObject()) {
                String message = detail.path("message").asText("");
                if (!message.isBlank()) {
                    return message;
                }
            }

            String message = root.path("message").asText("");
            if (!message.isBlank()) {
                return message;
            }

            return raw;
        } catch (Exception ignored) {
            return fallbackErrorMessage(statusCode);
        }
    }

    private String fallbackErrorMessage(int statusCode) {
        if (statusCode == 401 || statusCode == 403) {
            return "AI 对话服务鉴权失败，请检查服务配置";
        }
        if (statusCode == 429) {
            return "AI 对话服务调用过于频繁，请稍后重试";
        }
        if (statusCode >= 500) {
            return "AI 对话服务暂不可用，请稍后重试";
        }
        return "AI 对话服务调用失败，请稍后重试";
    }
}
