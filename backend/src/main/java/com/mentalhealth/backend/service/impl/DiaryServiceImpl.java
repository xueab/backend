package com.mentalhealth.backend.service.impl;

import com.mentalhealth.backend.common.BizException;
import com.mentalhealth.backend.common.PageResult;
import com.mentalhealth.backend.dto.CreateDiaryDTO;
import com.mentalhealth.backend.dto.DiaryQueryDTO;
import com.mentalhealth.backend.dto.UpdateDiaryDTO;
import com.mentalhealth.backend.entity.MoodDiary;
import com.mentalhealth.backend.mapper.MoodDiaryMapper;
import com.mentalhealth.backend.service.AiAnalysisService;
import com.mentalhealth.backend.service.DiaryService;
import com.mentalhealth.backend.vo.DiaryVO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class DiaryServiceImpl implements DiaryService {

    private final MoodDiaryMapper moodDiaryMapper;
    private final AiAnalysisService aiAnalysisService;

    @Override
    public Long create(Long userId, CreateDiaryDTO dto) {
        MoodDiary diary = new MoodDiary();
        diary.setUserId(userId);
        diary.setContent(dto.getContent().trim());
        diary.setMoodScore(dto.getMoodScore());
        diary.setTags(joinTags(dto.getTags()));
        moodDiaryMapper.insert(diary);
        return diary.getId();
    }

    @Override
    public PageResult<DiaryVO> page(Long userId, DiaryQueryDTO query) {
        int page = query.getPage() == null || query.getPage() < 1 ? 1 : query.getPage();
        int size = query.getSize() == null || query.getSize() < 1 ? 10 : query.getSize();
        if (size > 100) {
            size = 100;
        }
        validateScoreRange(query.getMinScore(), query.getMaxScore());

        LocalDateTime start = query.getStartDate() == null ? null : query.getStartDate().atStartOfDay();
        // 右开区间：endDate 当天的 23:59:59 也要命中，因此把 end 设为「次日 00:00」
        LocalDateTime endExclusive = query.getEndDate() == null
                ? null
                : query.getEndDate().plusDays(1).atStartOfDay();
        if (start != null && endExclusive != null && !start.isBefore(endExclusive)) {
            throw new BizException("开始时间必须早于结束时间");
        }

        long total = moodDiaryMapper.pageCount(userId, start, endExclusive,
                query.getMinScore(), query.getMaxScore());
        if (total == 0) {
            return PageResult.empty(page, size);
        }

        long offset = (long) (page - 1) * size;
        List<MoodDiary> rows = moodDiaryMapper.pageQuery(userId, start, endExclusive,
                query.getMinScore(), query.getMaxScore(), offset, size);
        List<DiaryVO> records = rows == null
                ? Collections.emptyList()
                : rows.stream().map(DiaryVO::fromEntity).collect(Collectors.toList());
        return PageResult.of(total, page, size, records);
    }

    @Override
    public DiaryVO detail(Long userId, Long diaryId) {
        MoodDiary diary = loadAndCheckOwner(userId, diaryId);
        return DiaryVO.fromEntity(diary);
    }

    @Override
    public DiaryVO update(Long userId, Long diaryId, UpdateDiaryDTO dto) {
        MoodDiary diary = loadAndCheckOwner(userId, diaryId);
        diary.setContent(dto.getContent().trim());
        diary.setMoodScore(dto.getMoodScore());
        diary.setTags(joinTags(dto.getTags()));
        moodDiaryMapper.update(diary);
        return DiaryVO.fromEntity(moodDiaryMapper.findById(diaryId));
    }

    @Override
    public void delete(Long userId, Long diaryId) {
        loadAndCheckOwner(userId, diaryId);
        moodDiaryMapper.deleteById(diaryId);
    }

    @Override
    public long todayCount(Long userId) {
        return moodDiaryMapper.countToday(userId);
    }

    @Override
    public DiaryVO aiAnalyze(Long userId, Long diaryId) {
        MoodDiary diary = loadAndCheckOwner(userId, diaryId);
        String analysis = aiAnalysisService.analyze(diary);
        moodDiaryMapper.updateAiAnalysis(diaryId, analysis);
        diary.setAiAnalysis(analysis);
        return DiaryVO.fromEntity(diary);
    }

    /* ========== 私有工具 ========== */

    /**
     * 加载日记并校验归属。
     * 不存在 → 404 语义；userId 不匹配 → 403。
     */
    private MoodDiary loadAndCheckOwner(Long userId, Long diaryId) {
        if (diaryId == null) {
            throw new BizException("日记 id 不能为空");
        }
        MoodDiary diary = moodDiaryMapper.findById(diaryId);
        if (diary == null) {
            throw new BizException(404, "日记不存在");
        }
        if (!userId.equals(diary.getUserId())) {
            throw new BizException(403, "无权限访问他人日记");
        }
        return diary;
    }

    private void validateScoreRange(Integer min, Integer max) {
        if (min != null && (min < 1 || min > 10)) {
            throw new BizException("minScore 必须在 1-10 之间");
        }
        if (max != null && (max < 1 || max > 10)) {
            throw new BizException("maxScore 必须在 1-10 之间");
        }
        if (min != null && max != null && min > max) {
            throw new BizException("minScore 不能大于 maxScore");
        }
    }

    private String joinTags(List<String> tags) {
        if (tags == null || tags.isEmpty()) {
            return null;
        }
        String joined = tags.stream()
                .filter(StringUtils::hasText)
                .map(t -> t.trim().replace(",", ""))
                .filter(t -> !t.isEmpty())
                .collect(Collectors.joining(","));
        return joined.isEmpty() ? null : joined;
    }
}
