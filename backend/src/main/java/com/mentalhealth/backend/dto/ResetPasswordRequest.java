package com.mentalhealth.backend.dto;

import lombok.Data;

@Data
public class ResetPasswordRequest {
    private String phone;
    private String password;
    private String code;
}
