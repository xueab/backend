package com.mentalhealth.backend.dto;

import java.util.List;

public record AiChatStreamRequest(
        Long sessionId,
        List<AiChatMessage> messages
) {
}
