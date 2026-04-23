package com.mentalhealth.backend.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.mentalhealth.backend.entity.MoodDiary;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 情绪日记视图对象。
 * 约定：
 * - date 为 yyyy-MM-dd HH:mm 的字符串，前端列表展示直接使用；
 * - score 即 mood_score，命名保持前端友好；
 * - tags 落库形式为逗号分隔字符串，此处还原为数组。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DiaryVO {

    private static final DateTimeFormatter DISPLAY_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");

    private Long id;

    /** 展示用的时间字符串 yyyy-MM-dd HH:mm */
    private String date;

    private String content;

    /** 心情分值 1-10，对应实体 moodScore */
    private Integer score;

    private List<String> tags;

    private String aiAnalysis;

    /** 原始时间，便于前端做二次排序/过滤 */
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createdAt;

    public static DiaryVO fromEntity(MoodDiary diary) {
        if (diary == null) {
            return null;
        }
        LocalDateTime created = diary.getCreatedAt();
        return DiaryVO.builder()
                .id(diary.getId())
                .date(created == null ? null : created.format(DISPLAY_FORMATTER))
                .content(diary.getContent())
                .score(diary.getMoodScore())
                .tags(splitTags(diary.getTags()))
                .aiAnalysis(diary.getAiAnalysis())
                .createdAt(created)
                .build();
    }

    private static List<String> splitTags(String raw) {
        if (raw == null || raw.isBlank()) {
            return Collections.emptyList();
        }
        return Arrays.stream(raw.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .collect(Collectors.toList());
    }
}
