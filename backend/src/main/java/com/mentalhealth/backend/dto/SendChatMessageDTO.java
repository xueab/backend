package com.mentalhealth.backend.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class SendChatMessageDTO {

    @NotBlank(message = "消息内容不能为空")
    @Size(max = 5000, message = "消息内容长度不能超过 5000 个字符")
    private String content;
}
