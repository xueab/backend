package com.mentalhealth.backend.vo;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 情绪汇总 VO。区间内：
 * - average：所有日记的平均分，保留 1 位小数；
 * - max / min：区间内出现的最高/最低单条分值；
 * - days：有记录的天数（去重）。
 * 区间内无记录时，全部字段置 0。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MoodSummaryVO {

    private Double average;

    private Integer max;

    private Integer min;

    private Integer days;
}
