package com.mentalhealth.backend.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.util.List;

/**
 * POST /api/diary 请求体：新建情绪日记。
 * tags 为可选的标签列表，Service 层会 join 为逗号分隔的字符串落库。
 */
@Data
public class CreateDiaryDTO {

    @NotBlank(message = "日记内容不能为空")
    @Size(max = 500, message = "日记内容长度不能超过 500 个字符")
    private String content;

    @NotNull(message = "心情分值不能为空")
    @Min(value = 1, message = "心情分值最小为 1")
    @Max(value = 10, message = "心情分值最大为 10")
    private Integer moodScore;

    private List<String> tags;
}
