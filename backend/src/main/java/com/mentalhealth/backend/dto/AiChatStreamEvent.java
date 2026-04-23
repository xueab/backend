package com.mentalhealth.backend.dto;

public record AiChatStreamEvent(
        String type,
        String content,
        String requestId,
        String errorCode,
        String message,
        Boolean retryable
) {
}
