package com.mentalhealth.backend.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class ChatSession {
    private Long id;
    private Long userId;
    private String title;
    /** 状态: 0 进行中, 1 已结束 */
    private Integer status;
    private String reportUrl;
    private LocalDateTime createdAt;
}
