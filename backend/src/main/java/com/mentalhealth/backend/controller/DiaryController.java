package com.mentalhealth.backend.controller;

import com.mentalhealth.backend.common.PageResult;
import com.mentalhealth.backend.common.Result;
import com.mentalhealth.backend.common.SecurityUtils;
import com.mentalhealth.backend.dto.CreateDiaryDTO;
import com.mentalhealth.backend.dto.DiaryQueryDTO;
import com.mentalhealth.backend.dto.UpdateDiaryDTO;
import com.mentalhealth.backend.service.DiaryService;
import com.mentalhealth.backend.vo.DiaryVO;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * 情绪日记模块控制器。
 * 路径前缀 /api/diary，SecurityConfig 的 anyRequest().authenticated() 保证必须携带 JWT，
 * userId 统一通过 {@link SecurityUtils} 获取，禁止从前端传入以防越权。
 */
@RestController
@RequestMapping("/api/diary")
@RequiredArgsConstructor
public class DiaryController {

    private final DiaryService diaryService;

    /** 新建日记，返回 { diaryId } 结构。 */
    @PostMapping
    public Result<Map<String, Long>> create(@RequestBody @Valid CreateDiaryDTO dto) {
        Long userId = SecurityUtils.getCurrentUserId();
        Long diaryId = diaryService.create(userId, dto);
        return Result.success(Map.of("diaryId", diaryId));
    }

    /** 分页查询当前用户的日记。 */
    @GetMapping("/page")
    public Result<PageResult<DiaryVO>> page(DiaryQueryDTO query) {
        Long userId = SecurityUtils.getCurrentUserId();
        return Result.success(diaryService.page(userId, query));
    }

    /** 今日日记数量，返回 { count }。 */
    @GetMapping("/today-count")
    public Result<Map<String, Long>> todayCount() {
        Long userId = SecurityUtils.getCurrentUserId();
        return Result.success(Map.of("count", diaryService.todayCount(userId)));
    }

    /** 详情。 */
    @GetMapping("/{id}")
    public Result<DiaryVO> detail(@PathVariable("id") Long id) {
        Long userId = SecurityUtils.getCurrentUserId();
        return Result.success(diaryService.detail(userId, id));
    }

    /** 更新。 */
    @PutMapping("/{id}")
    public Result<DiaryVO> update(@PathVariable("id") Long id,
                                  @RequestBody @Valid UpdateDiaryDTO dto) {
        Long userId = SecurityUtils.getCurrentUserId();
        return Result.success(diaryService.update(userId, id, dto));
    }

    /** 删除。 */
    @DeleteMapping("/{id}")
    public Result<?> delete(@PathVariable("id") Long id) {
        Long userId = SecurityUtils.getCurrentUserId();
        diaryService.delete(userId, id);
        return Result.success("删除成功");
    }

    /** 触发 AI 分析，返回写回后的日记 VO。 */
    @PostMapping("/{id}/ai-analysis")
    public Result<DiaryVO> aiAnalyze(@PathVariable("id") Long id) {
        Long userId = SecurityUtils.getCurrentUserId();
        return Result.success(diaryService.aiAnalyze(userId, id));
    }
}
