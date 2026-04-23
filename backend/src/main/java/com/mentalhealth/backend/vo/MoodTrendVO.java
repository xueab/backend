package com.mentalhealth.backend.vo;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * 情绪趋势 VO。
 * dates 与 scores 下标一一对应；某天没有记录时，score 为 null，
 * 前端可据此决定「画断点」还是「跳过」。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MoodTrendVO {

    /** MM-dd 格式的日期标签，按时间升序 */
    private List<String> dates;

    /** 对应日期的平均分（保留 1 位小数），无记录为 null */
    private List<Double> scores;
}
