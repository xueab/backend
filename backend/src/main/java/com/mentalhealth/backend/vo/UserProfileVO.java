package com.mentalhealth.backend.vo;

import com.mentalhealth.backend.entity.User;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * GET/PUT /api/user/profile 的响应体：
 * 明确只暴露前端需要的字段，password 等敏感字段绝不外泄。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserProfileVO {
    private Long userId;
    private String phone;
    private String nickname;
    private String avatar;
    private LocalDateTime createdAt;

    public static UserProfileVO fromEntity(User user) {
        if (user == null) {
            return null;
        }
        return UserProfileVO.builder()
                .userId(user.getId())
                .phone(user.getPhone())
                .nickname(user.getNickname())
                .avatar(user.getAvatar())
                .createdAt(user.getCreatedAt())
                .build();
    }
}
