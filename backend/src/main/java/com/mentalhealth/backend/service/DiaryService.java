package com.mentalhealth.backend.service;

import com.mentalhealth.backend.common.PageResult;
import com.mentalhealth.backend.dto.CreateDiaryDTO;
import com.mentalhealth.backend.dto.DiaryQueryDTO;
import com.mentalhealth.backend.dto.UpdateDiaryDTO;
import com.mentalhealth.backend.vo.DiaryVO;

/**
 * 情绪日记业务接口。
 * 所有方法的「当前用户」均显式以 userId 入参，由 Controller 从 SecurityContext 获取，
 * 以避免业务层直接依赖 Spring Security 上下文，便于单测。
 */
public interface DiaryService {

    /** 新建日记，返回主键 id。 */
    Long create(Long userId, CreateDiaryDTO dto);

    /** 按条件分页。 */
    PageResult<DiaryVO> page(Long userId, DiaryQueryDTO query);

    /** 详情；越权访问将抛出 403 业务异常。 */
    DiaryVO detail(Long userId, Long diaryId);

    /** 更新；越权访问将抛出 403 业务异常。 */
    DiaryVO update(Long userId, Long diaryId, UpdateDiaryDTO dto);

    /** 删除；越权访问将抛出 403 业务异常。 */
    void delete(Long userId, Long diaryId);

    /** 当前用户当天日记数量。 */
    long todayCount(Long userId);

    /** 触发 AI 分析并写回 aiAnalysis 字段，返回更新后的 VO。 */
    DiaryVO aiAnalyze(Long userId, Long diaryId);
}
