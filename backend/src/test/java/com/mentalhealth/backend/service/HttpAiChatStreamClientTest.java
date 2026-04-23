package com.mentalhealth.backend.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.mentalhealth.backend.common.BizException;
import com.mentalhealth.backend.config.AiServiceProperties;
import com.mentalhealth.backend.service.ai.HttpAiChatStreamClient;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class HttpAiChatStreamClientTest {

    private HttpServer server;
    private HttpAiChatStreamClient client;

    @BeforeEach
    void setUp() throws IOException {
        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.start();

        AiServiceProperties properties = new AiServiceProperties();
        properties.setBaseUrl("http://localhost:" + server.getAddress().getPort());
        properties.setChatStreamPath("/internal/v1/chat/stream");
        properties.setStreamReadTimeoutMs(3000);

        client = new HttpAiChatStreamClient(
                HttpClient.newHttpClient(),
                new ObjectMapper(),
                properties
        );
    }

    @AfterEach
    void tearDown() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void streamChat_whenUpstreamReturnsStructuredError_throwsDetailedBizException() {
        server.createContext("/internal/v1/chat/stream", new JsonErrorHandler());

        BizException ex = assertThrows(BizException.class, () ->
                client.streamChat(new com.mentalhealth.backend.dto.AiChatStreamRequest(1L, java.util.List.of()), event -> {
                })
        );

        assertEquals(503, ex.getCode());
        assertEquals("DeepSeek API Key 未配置", ex.getMessage());
    }

    private static class JsonErrorHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            byte[] body = """
                    {
                      "detail": {
                        "errorCode": "MISSING_API_KEY",
                        "message": "DeepSeek API Key 未配置",
                        "retryable": false
                      }
                    }
                    """.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(503, body.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(body);
            }
        }
    }
}
