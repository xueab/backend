package com.mentalhealth.backend.dto;

import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.time.LocalDate;

/**
 * GET /api/diary/page 的查询条件载体。
 * 日期按 yyyy-MM-dd 解析；Service 层会把 endDate 转换为「次日 00:00」做左闭右开比较。
 */
@Data
public class DiaryQueryDTO {

    private Integer page = 1;
    private Integer size = 10;

    @DateTimeFormat(iso = DateTimeFormat.ISO.DATE)
    private LocalDate startDate;

    @DateTimeFormat(iso = DateTimeFormat.ISO.DATE)
    private LocalDate endDate;

    private Integer minScore;
    private Integer maxScore;
}
