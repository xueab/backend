package com.mentalhealth.backend.service;

import com.mentalhealth.backend.entity.MoodDiary;

/**
 * AI 情绪分析服务抽象。
 * 由具体实现决定是调用远程 ai-service 还是其它模型提供方。
 */
public interface AiAnalysisService {

    /**
     * 根据日记内容与分值，返回一段分析文案。
     */
    String analyze(MoodDiary diary);
}
