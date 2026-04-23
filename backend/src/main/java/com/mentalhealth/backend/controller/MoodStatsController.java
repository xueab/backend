package com.mentalhealth.backend.controller;

import com.mentalhealth.backend.common.Result;
import com.mentalhealth.backend.common.SecurityUtils;
import com.mentalhealth.backend.service.MoodStatsService;
import com.mentalhealth.backend.vo.MoodInsightVO;
import com.mentalhealth.backend.vo.MoodSummaryVO;
import com.mentalhealth.backend.vo.MoodTrendVO;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * 情绪统计模块。所有接口需登录，userId 通过 {@link SecurityUtils} 获取；
 * range 支持 7 / 14 / 30，其它值由 Service 抛 400。
 */
@RestController
@RequestMapping("/api/mood")
@RequiredArgsConstructor
public class MoodStatsController {

    private final MoodStatsService moodStatsService;

    /** 趋势折线数据：dates 与 scores 下标一一对应。 */
    @GetMapping("/trend")
    public Result<MoodTrendVO> trend(@RequestParam(value = "range", defaultValue = "7") Integer range) {
        Long userId = SecurityUtils.getCurrentUserId();
        return Result.success(moodStatsService.trend(userId, range));
    }

    /** 汇总：平均 / 最高 / 最低 / 有记录天数。 */
    @GetMapping("/summary")
    public Result<MoodSummaryVO> summary(@RequestParam(value = "range", defaultValue = "7") Integer range) {
        Long userId = SecurityUtils.getCurrentUserId();
        return Result.success(moodStatsService.summary(userId, range));
    }

    /** 文字建议（先硬编码，后续可切换 AI 产出）。 */
    @GetMapping("/insights")
    public Result<MoodInsightVO> insights(@RequestParam(value = "range", defaultValue = "7") Integer range) {
        Long userId = SecurityUtils.getCurrentUserId();
        return Result.success(moodStatsService.insights(userId, range));
    }
}
