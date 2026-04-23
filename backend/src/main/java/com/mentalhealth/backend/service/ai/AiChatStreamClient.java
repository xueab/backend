package com.mentalhealth.backend.service.ai;

import com.mentalhealth.backend.dto.AiChatStreamEvent;
import com.mentalhealth.backend.dto.AiChatStreamRequest;

import java.util.function.Consumer;

public interface AiChatStreamClient {

    void streamChat(AiChatStreamRequest request, Consumer<AiChatStreamEvent> eventConsumer);
}
