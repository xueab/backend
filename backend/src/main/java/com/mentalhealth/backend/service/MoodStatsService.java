package com.mentalhealth.backend.service;

import com.mentalhealth.backend.vo.MoodInsightVO;
import com.mentalhealth.backend.vo.MoodSummaryVO;
import com.mentalhealth.backend.vo.MoodTrendVO;

/**
 * 情绪统计 Service。
 * 所有接口以「最近 N 天（含今天）」为口径：range = 7/14/30，其它值视为非法参数。
 */
public interface MoodStatsService {

    MoodTrendVO trend(Long userId, Integer range);

    MoodSummaryVO summary(Long userId, Integer range);

    MoodInsightVO insights(Long userId, Integer range);
}
