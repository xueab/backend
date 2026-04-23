package com.mentalhealth.backend.controller;

import com.mentalhealth.backend.common.Result;
import com.mentalhealth.backend.common.SecurityUtils;
import com.mentalhealth.backend.dto.CreateChatSessionDTO;
import com.mentalhealth.backend.dto.SendChatMessageDTO;
import com.mentalhealth.backend.service.ChatService;
import com.mentalhealth.backend.vo.ChatMessageVO;
import com.mentalhealth.backend.vo.ChatSessionVO;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
public class ChatController {

    private final ChatService chatService;

    @PostMapping("/sessions")
    public Result<Map<String, Long>> createSession(@RequestBody(required = false) @Valid CreateChatSessionDTO dto) {
        Long userId = SecurityUtils.getCurrentUserId();
        Long sessionId = chatService.createSession(userId, dto);
        return Result.success(Map.of("sessionId", sessionId));
    }

    @GetMapping("/sessions")
    public Result<List<ChatSessionVO>> listSessions() {
        Long userId = SecurityUtils.getCurrentUserId();
        return Result.success(chatService.listSessions(userId));
    }

    @GetMapping("/sessions/{id}/messages")
    public Result<List<ChatMessageVO>> listMessages(@PathVariable("id") Long id) {
        Long userId = SecurityUtils.getCurrentUserId();
        return Result.success(chatService.listMessages(userId, id));
    }

    @PostMapping(value = "/sessions/{id}/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamReply(@PathVariable("id") Long id,
                                  @RequestBody @Valid SendChatMessageDTO dto) {
        Long userId = SecurityUtils.getCurrentUserId();
        return chatService.streamReply(userId, id, dto);
    }
}
