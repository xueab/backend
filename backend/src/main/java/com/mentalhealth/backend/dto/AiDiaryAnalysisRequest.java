package com.mentalhealth.backend.dto;

public record AiDiaryAnalysisRequest(
        Long diaryId,
        String content,
        Integer moodScore
) {
}
