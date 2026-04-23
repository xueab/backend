package com.mentalhealth.backend.service;

import com.mentalhealth.backend.entity.MoodDiary;
import org.springframework.stereotype.Service;

/**
 * AI 情绪分析服务。
 * 当前为 Mock 实现，后续接入真实大模型时只需替换方法体（或改为 interface + 多实现）。
 */
@Service
public class AiAnalysisService {

    /**
     * 根据日记内容与分值，返回一段分析文案。
     * Mock 策略：按 moodScore 的 [1-3] / [4-6] / [7-10] 三档返回不同的固定文案，
     * 便于前端联调时能看到差异化展示效果。
     */
    public String analyze(MoodDiary diary) {
        Integer score = diary == null ? null : diary.getMoodScore();
        if (score == null) {
            return "已收到你的情绪记录，我正在仔细阅读这段文字，并会陪你慢慢梳理其中的感受。";
        }
        if (score <= 3) {
            return "从你的文字中能感受到明显的低落与压力。请记得：情绪只是此刻的一种状态，并不定义你。"
                    + "建议先给自己 10 分钟的深呼吸或散步时间，再尝试把一个具体的小困扰拆解成可行动的下一步。";
        }
        if (score <= 6) {
            return "你的情绪处于中等波动区间，日常的琐碎可能正在悄悄消耗能量。"
                    + "可以尝试列出今天让你感到「略有成就」的三件小事，用正反馈抵消部分负面干扰。";
        }
        return "你今天整体状态不错，记录下这份积极的能量非常有价值。"
                + "可以进一步回忆是哪些行为带来了愉悦，沉淀成你专属的「情绪急救工具箱」。";
    }
}
