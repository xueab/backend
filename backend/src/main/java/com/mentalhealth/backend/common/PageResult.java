package com.mentalhealth.backend.common;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Collections;
import java.util.List;

/**
 * 通用分页响应体。
 * 与 {@link Result} 组合使用：Result&lt;PageResult&lt;DiaryVO&gt;&gt;。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PageResult<T> {

    /** 总记录数 */
    private long total;
    /** 当前页码，从 1 开始 */
    private long page;
    /** 每页大小 */
    private long size;
    /** 当前页数据 */
    private List<T> records;

    public static <T> PageResult<T> empty(long page, long size) {
        return PageResult.<T>builder()
                .total(0L)
                .page(page)
                .size(size)
                .records(Collections.emptyList())
                .build();
    }

    public static <T> PageResult<T> of(long total, long page, long size, List<T> records) {
        return PageResult.<T>builder()
                .total(total)
                .page(page)
                .size(size)
                .records(records)
                .build();
    }
}
