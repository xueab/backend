package com.mentalhealth.backend.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.mentalhealth.backend.entity.ChatMessage;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ChatMessageVO {

    private Long id;
    private String role;
    private String content;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createdAt;

    public static ChatMessageVO fromEntity(ChatMessage message) {
        if (message == null) {
            return null;
        }
        return ChatMessageVO.builder()
                .id(message.getId())
                .role(message.getRole())
                .content(message.getContent())
                .createdAt(message.getCreatedAt())
                .build();
    }
}
