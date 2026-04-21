package com.mentalhealth.backend.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class MoodDiary {
    private Long id;
    private Long userId;
    private String content;
    /** 情绪分值 1-10 */
    private Integer moodScore;
    /** 情绪标签，逗号分隔 */
    private String tags;
    private String aiAnalysis;
    private LocalDateTime createdAt;
}
