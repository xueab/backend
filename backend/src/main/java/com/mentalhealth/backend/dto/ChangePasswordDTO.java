package com.mentalhealth.backend.dto;

import lombok.Data;

/**
 * PUT /api/user/password 的请求体。
 */
@Data
public class ChangePasswordDTO {
    private String oldPassword;
    private String newPassword;
}
