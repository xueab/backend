package com.mentalhealth.backend.service;

import com.mentalhealth.backend.dto.CreateChatSessionDTO;
import com.mentalhealth.backend.dto.SendChatMessageDTO;
import com.mentalhealth.backend.vo.ChatMessageVO;
import com.mentalhealth.backend.vo.ChatSessionVO;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.List;

public interface ChatService {

    Long createSession(Long userId, CreateChatSessionDTO dto);

    List<ChatSessionVO> listSessions(Long userId);

    List<ChatMessageVO> listMessages(Long userId, Long sessionId);

    SseEmitter streamReply(Long userId, Long sessionId, SendChatMessageDTO dto);
}
