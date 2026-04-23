package com.mentalhealth.backend.service;

import com.mentalhealth.backend.common.BizException;
import com.mentalhealth.backend.entity.MoodDiary;
import com.mentalhealth.backend.mapper.MoodDiaryMapper;
import com.mentalhealth.backend.service.impl.DiaryServiceImpl;
import com.mentalhealth.backend.vo.DiaryVO;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class DiaryServiceImplTest {

    @Mock
    private MoodDiaryMapper moodDiaryMapper;

    @Mock
    private AiAnalysisService aiAnalysisService;

    @InjectMocks
    private DiaryServiceImpl diaryService;

    @Test
    void aiAnalyze_success_persistsAnalysis() {
        MoodDiary diary = sampleDiary();
        when(moodDiaryMapper.findById(100L)).thenReturn(diary);
        when(aiAnalysisService.analyze(diary)).thenReturn("AI 分析内容");

        DiaryVO result = diaryService.aiAnalyze(1L, 100L);

        assertEquals("AI 分析内容", result.getAiAnalysis());
        verify(moodDiaryMapper).updateAiAnalysis(100L, "AI 分析内容");
    }

    @Test
    void aiAnalyze_whenAiFails_doesNotOverwriteStoredAnalysis() {
        MoodDiary diary = sampleDiary();
        when(moodDiaryMapper.findById(100L)).thenReturn(diary);
        when(aiAnalysisService.analyze(diary)).thenThrow(new BizException(503, "AI 服务暂不可用"));

        BizException ex = assertThrows(BizException.class, () -> diaryService.aiAnalyze(1L, 100L));

        assertEquals(503, ex.getCode());
        verify(moodDiaryMapper, never()).updateAiAnalysis(100L, "AI 分析内容");
    }

    private MoodDiary sampleDiary() {
        MoodDiary diary = new MoodDiary();
        diary.setId(100L);
        diary.setUserId(1L);
        diary.setContent("今天感觉压力有点大，但还是完成了任务。");
        diary.setMoodScore(5);
        diary.setCreatedAt(LocalDateTime.now());
        diary.setAiAnalysis("旧分析");
        return diary;
    }
}
