package com.mentalhealth.backend.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.mentalhealth.backend.entity.ChatSession;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ChatSessionVO {

    private Long id;
    private String title;
    private Integer status;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createdAt;

    public static ChatSessionVO fromEntity(ChatSession session) {
        if (session == null) {
            return null;
        }
        return ChatSessionVO.builder()
                .id(session.getId())
                .title(session.getTitle())
                .status(session.getStatus())
                .createdAt(session.getCreatedAt())
                .build();
    }
}
