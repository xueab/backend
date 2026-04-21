package com.mentalhealth.backend.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class ChatMessage {
    private Long id;
    private Long sessionId;
    /** 角色: user / assistant */
    private String role;
    private String content;
    private LocalDateTime createdAt;
}
