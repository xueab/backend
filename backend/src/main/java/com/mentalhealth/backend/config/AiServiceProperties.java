package com.mentalhealth.backend.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Getter
@Setter
@ConfigurationProperties(prefix = "ai-service")
public class AiServiceProperties {

    /**
     * ai-service 的基础地址，例如 http://localhost:8001
     */
    private String baseUrl = "http://localhost:8001";

    /**
     * 日记分析接口路径。
     */
    private String analyzePath = "/internal/v1/mood/analyze";

    /**
     * AI 对话流式接口路径。
     */
    private String chatStreamPath = "/internal/v1/chat/stream";

    /**
     * backend 调 ai-service 时可选的内部鉴权 token。
     */
    private String internalToken = "";

    private int connectTimeoutMs = 3000;

    private int readTimeoutMs = 30000;

    private int streamReadTimeoutMs = 60000;
}
