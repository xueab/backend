package com.mentalhealth.backend.service;

import com.mentalhealth.backend.dto.AiChatMessage;
import com.mentalhealth.backend.dto.AiChatStreamEvent;
import com.mentalhealth.backend.dto.AiChatStreamRequest;
import com.mentalhealth.backend.dto.SendChatMessageDTO;
import com.mentalhealth.backend.entity.ChatMessage;
import com.mentalhealth.backend.entity.ChatSession;
import com.mentalhealth.backend.mapper.ChatMessageMapper;
import com.mentalhealth.backend.mapper.ChatSessionMapper;
import com.mentalhealth.backend.service.ai.AiChatStreamClient;
import com.mentalhealth.backend.service.impl.ChatServiceImpl;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.time.LocalDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ChatServiceImplTest {

    @Mock
    private ChatSessionMapper chatSessionMapper;

    @Mock
    private ChatMessageMapper chatMessageMapper;

    @Mock
    private AiChatStreamClient aiChatStreamClient;

    @Test
    void streamReply_whenDoneEventArrives_persistsAssistantMessage() {
        ChatServiceImpl service = new ChatServiceImpl(
                chatSessionMapper,
                chatMessageMapper,
                aiChatStreamClient,
                Runnable::run
        );
        SendChatMessageDTO dto = new SendChatMessageDTO();
        dto.setContent("我最近有点焦虑。");

        when(chatSessionMapper.findById(1L)).thenReturn(sampleSession());
        when(chatMessageMapper.findBySessionId(1L)).thenReturn(List.of(sampleUserMessage()));
        doAnswer(invocation -> {
            AiChatStreamRequest request = invocation.getArgument(0);
            assertTrue(request.messages().stream().map(AiChatMessage::content).anyMatch("我最近有点焦虑。"::equals));
            invocation.<java.util.function.Consumer<AiChatStreamEvent>>getArgument(1)
                    .accept(new AiChatStreamEvent("delta", "先深呼吸一下，", "req-1", null, null, null));
            invocation.<java.util.function.Consumer<AiChatStreamEvent>>getArgument(1)
                    .accept(new AiChatStreamEvent("delta", "你已经很努力了。", "req-1", null, null, null));
            invocation.<java.util.function.Consumer<AiChatStreamEvent>>getArgument(1)
                    .accept(new AiChatStreamEvent("done", "", "req-1", null, null, null));
            return null;
        }).when(aiChatStreamClient).streamChat(any(AiChatStreamRequest.class), any());

        SseEmitter emitter = service.streamReply(7L, 1L, dto);

        assertNotNull(emitter);
        ArgumentCaptor<ChatMessage> captor = ArgumentCaptor.forClass(ChatMessage.class);
        verify(chatMessageMapper, times(2)).insert(captor.capture());
        List<ChatMessage> inserted = captor.getAllValues();
        assertTrue(inserted.stream().anyMatch(msg -> "assistant".equals(msg.getRole())
                && "先深呼吸一下，你已经很努力了。".equals(msg.getContent())));
    }

    @Test
    void streamReply_whenOnlyErrorEventArrives_doesNotPersistAssistantMessage() {
        ChatServiceImpl service = new ChatServiceImpl(
                chatSessionMapper,
                chatMessageMapper,
                aiChatStreamClient,
                Runnable::run
        );
        SendChatMessageDTO dto = new SendChatMessageDTO();
        dto.setContent("我很难过。");

        when(chatSessionMapper.findById(1L)).thenReturn(sampleSession());
        when(chatMessageMapper.findBySessionId(1L)).thenReturn(List.of(sampleUserMessage()));
        doAnswer(invocation -> {
            invocation.<java.util.function.Consumer<AiChatStreamEvent>>getArgument(1)
                    .accept(new AiChatStreamEvent("error", "", "req-2", "UPSTREAM", "失败", true));
            return null;
        }).when(aiChatStreamClient).streamChat(any(AiChatStreamRequest.class), any());

        service.streamReply(7L, 1L, dto);

        verify(chatMessageMapper, times(1)).insert(any(ChatMessage.class));
        verify(chatMessageMapper, never()).insert(argThat(msg ->
                "assistant".equals(msg.getRole()) && msg.getContent() != null && !msg.getContent().isEmpty()));
    }

    @Test
    void streamReply_filtersInvalidHistoryBeforeCallingAiService() {
        ChatServiceImpl service = new ChatServiceImpl(
                chatSessionMapper,
                chatMessageMapper,
                aiChatStreamClient,
                Runnable::run
        );
        SendChatMessageDTO dto = new SendChatMessageDTO();
        dto.setContent("请继续陪我聊聊。");

        ChatMessage invalidRole = new ChatMessage();
        invalidRole.setSessionId(1L);
        invalidRole.setRole(" system ");
        invalidRole.setContent("历史中的异常角色");

        ChatMessage blankContent = new ChatMessage();
        blankContent.setSessionId(1L);
        blankContent.setRole("assistant");
        blankContent.setContent("   ");

        when(chatSessionMapper.findById(1L)).thenReturn(sampleSession());
        when(chatMessageMapper.findBySessionId(1L)).thenReturn(List.of(sampleUserMessage(), invalidRole, blankContent));
        doAnswer(invocation -> {
            AiChatStreamRequest request = invocation.getArgument(0);
            assertEquals(2, request.messages().size());
            assertEquals("user", request.messages().get(0).role());
            assertEquals("user", request.messages().get(1).role());
            invocation.<java.util.function.Consumer<AiChatStreamEvent>>getArgument(1)
                    .accept(new AiChatStreamEvent("done", "", "req-3", null, null, null));
            return null;
        }).when(aiChatStreamClient).streamChat(any(AiChatStreamRequest.class), any());

        service.streamReply(7L, 1L, dto);

        verify(aiChatStreamClient).streamChat(any(AiChatStreamRequest.class), any());
    }

    private ChatSession sampleSession() {
        ChatSession session = new ChatSession();
        session.setId(1L);
        session.setUserId(7L);
        session.setTitle("新对话");
        session.setStatus(0);
        session.setCreatedAt(LocalDateTime.now());
        return session;
    }

    private ChatMessage sampleUserMessage() {
        ChatMessage message = new ChatMessage();
        message.setId(11L);
        message.setSessionId(1L);
        message.setRole("user");
        message.setContent("我最近有点焦虑。");
        message.setCreatedAt(LocalDateTime.now());
        return message;
    }
}
