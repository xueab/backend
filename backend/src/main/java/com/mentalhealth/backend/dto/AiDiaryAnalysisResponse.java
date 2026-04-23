package com.mentalhealth.backend.dto;

public record AiDiaryAnalysisResponse(
        String analysisText,
        String model,
        String requestId
) {
}
