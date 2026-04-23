package com.mentalhealth.backend.dto;

import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class CreateChatSessionDTO {

    @Size(max = 50, message = "会话标题长度不能超过 50 个字符")
    private String title;
}
