package com.mentalhealth.backend.service.impl;

import com.mentalhealth.backend.common.BizException;
import com.mentalhealth.backend.dto.AiChatMessage;
import com.mentalhealth.backend.dto.AiChatStreamEvent;
import com.mentalhealth.backend.dto.AiChatStreamRequest;
import com.mentalhealth.backend.dto.CreateChatSessionDTO;
import com.mentalhealth.backend.dto.SendChatMessageDTO;
import com.mentalhealth.backend.entity.ChatMessage;
import com.mentalhealth.backend.entity.ChatSession;
import com.mentalhealth.backend.mapper.ChatMessageMapper;
import com.mentalhealth.backend.mapper.ChatSessionMapper;
import com.mentalhealth.backend.service.ChatService;
import com.mentalhealth.backend.service.ai.AiChatStreamClient;
import com.mentalhealth.backend.vo.ChatMessageVO;
import com.mentalhealth.backend.vo.ChatSessionVO;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ChatServiceImpl implements ChatService {

    private static final long SSE_TIMEOUT_MS = 0L;
    private static final String DEFAULT_TITLE = "新对话";

    private final ChatSessionMapper chatSessionMapper;
    private final ChatMessageMapper chatMessageMapper;
    private final AiChatStreamClient aiChatStreamClient;

    @Qualifier("chatSseExecutor")
    private final Executor chatSseExecutor;

    @Override
    public Long createSession(Long userId, CreateChatSessionDTO dto) {
        ChatSession session = new ChatSession();
        session.setUserId(userId);
        session.setTitle(resolveTitle(dto));
        session.setStatus(0);
        chatSessionMapper.insert(session);
        return session.getId();
    }

    @Override
    public List<ChatSessionVO> listSessions(Long userId) {
        return chatSessionMapper.findByUserId(userId).stream()
                .map(ChatSessionVO::fromEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<ChatMessageVO> listMessages(Long userId, Long sessionId) {
        loadAndCheckOwner(userId, sessionId);
        return chatMessageMapper.findBySessionId(sessionId).stream()
                .map(ChatMessageVO::fromEntity)
                .collect(Collectors.toList());
    }

    @Override
    public SseEmitter streamReply(Long userId, Long sessionId, SendChatMessageDTO dto) {
        ChatSession session = loadAndCheckOwner(userId, sessionId);
        String content = dto.getContent().trim();

        ChatMessage userMessage = new ChatMessage();
        userMessage.setSessionId(session.getId());
        userMessage.setRole("user");
        userMessage.setContent(content);
        chatMessageMapper.insert(userMessage);

        List<AiChatMessage> history = chatMessageMapper.findBySessionId(sessionId).stream()
                .filter(message -> StringUtils.hasText(message.getRole()) && StringUtils.hasText(message.getContent()))
                .map(message -> new AiChatMessage(normalizeRole(message.getRole()), message.getContent().trim()))
                .collect(Collectors.toList());

        SseEmitter emitter = new SseEmitter(SSE_TIMEOUT_MS);
        emitter.onTimeout(emitter::complete);

        chatSseExecutor.execute(() -> streamAssistantReply(sessionId, history, emitter));
        return emitter;
    }

    private void streamAssistantReply(Long sessionId, List<AiChatMessage> history, SseEmitter emitter) {
        StringBuilder assistantContent = new StringBuilder();
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicBoolean completed = new AtomicBoolean(false);

        try {
            aiChatStreamClient.streamChat(new AiChatStreamRequest(sessionId, history), event -> {
                handleStreamEvent(emitter, assistantContent, failed, completed, event);
            });

            if (completed.get() && !failed.get() && StringUtils.hasText(assistantContent.toString())) {
                ChatMessage assistantMessage = new ChatMessage();
                assistantMessage.setSessionId(sessionId);
                assistantMessage.setRole("assistant");
                assistantMessage.setContent(assistantContent.toString());
                chatMessageMapper.insert(assistantMessage);
            }
        } catch (Exception e) {
            failed.set(true);
            sendSafely(emitter, "error", Map.of(
                    "type", "error",
                    "message", resolveErrorMessage(e)
            ));
        } finally {
            emitter.complete();
        }
    }

    private void handleStreamEvent(SseEmitter emitter,
                                   StringBuilder assistantContent,
                                   AtomicBoolean failed,
                                   AtomicBoolean completed,
                                   AiChatStreamEvent event) {
        String eventType = StringUtils.hasText(event.type()) ? event.type() : "delta";

        if ("delta".equals(eventType) && StringUtils.hasText(event.content())) {
            assistantContent.append(event.content());
        }
        if ("error".equals(eventType)) {
            failed.set(true);
        }
        if ("done".equals(eventType)) {
            completed.set(true);
        }

        sendSafely(emitter, eventType, Map.of(
                "type", eventType,
                "content", event.content() == null ? "" : event.content(),
                "requestId", event.requestId() == null ? "" : event.requestId(),
                "errorCode", event.errorCode() == null ? "" : event.errorCode(),
                "message", event.message() == null ? "" : event.message(),
                "retryable", event.retryable() != null && event.retryable()
        ));
    }

    private void sendSafely(SseEmitter emitter, String eventName, Object data) {
        try {
            emitter.send(SseEmitter.event()
                    .name(eventName)
                    .data(data, MediaType.APPLICATION_JSON));
        } catch (IOException e) {
            throw new BizException(499, "客户端已断开连接");
        }
    }

    private ChatSession loadAndCheckOwner(Long userId, Long sessionId) {
        if (sessionId == null) {
            throw new BizException("会话 id 不能为空");
        }
        ChatSession session = chatSessionMapper.findById(sessionId);
        if (session == null) {
            throw new BizException(404, "会话不存在");
        }
        if (!userId.equals(session.getUserId())) {
            throw new BizException(403, "无权限访问该会话");
        }
        return session;
    }

    private String resolveTitle(CreateChatSessionDTO dto) {
        if (dto == null || !StringUtils.hasText(dto.getTitle())) {
            return DEFAULT_TITLE;
        }
        return dto.getTitle().trim();
    }

    private String normalizeRole(String role) {
        String normalized = role == null ? "" : role.trim().toLowerCase();
        if ("assistant".equals(normalized)) {
            return "assistant";
        }
        return "user";
    }

    private String resolveErrorMessage(Exception e) {
        if (e instanceof BizException bizException && StringUtils.hasText(bizException.getMessage())) {
            return bizException.getMessage();
        }
        return "AI 对话生成失败，请稍后重试";
    }
}
