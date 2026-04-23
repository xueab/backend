package com.mentalhealth.backend.dto;

import lombok.Data;

/**
 * PUT /api/user/profile 的请求体。
 * 目前仅允许修改昵称，后续如放开更多字段在此扩展即可。
 */
@Data
public class UpdateProfileDTO {
    private String nickname;
}
