package com.mentalhealth.backend.common;

import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;

/**
 * 当前登录用户工具类。
 * 约定：{@link com.mentalhealth.backend.config.JwtAuthenticationFilter} 在鉴权成功后，
 * 会把 userId 设为 Authentication.principal，phone 设为 credentials。
 * 所有「需要 userId」的业务必须通过此工具获取，禁止从请求参数传入。
 */
public final class SecurityUtils {

    private SecurityUtils() {
    }

    public static Long getCurrentUserId() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication == null || !authentication.isAuthenticated()) {
            throw new BizException(401, "未登录或登录已过期");
        }
        Object principal = authentication.getPrincipal();
        if (principal instanceof Long userId) {
            return userId;
        }
        throw new BizException(401, "登录凭证无效");
    }

    public static String getCurrentPhone() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication == null) {
            return null;
        }
        Object credentials = authentication.getCredentials();
        return credentials == null ? null : credentials.toString();
    }
}
