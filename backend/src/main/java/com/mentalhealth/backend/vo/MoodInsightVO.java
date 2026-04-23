package com.mentalhealth.backend.vo;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * 情绪洞察 VO。tips 为按顺序展示的一组文字建议，
 * 当前基于阈值硬编码模板生成，后续可切换到 AI 输出。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MoodInsightVO {

    private List<String> tips;
}
