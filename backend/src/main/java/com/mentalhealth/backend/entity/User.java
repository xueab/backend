package com.mentalhealth.backend.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class User {
    private Long id;
    /** 对应数据库 username 列，业务上是手机号 */
    private String phone;
    private String password;
    private String nickname;
    private String avatar;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
