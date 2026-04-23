package com.mentalhealth.backend.service;

import com.mentalhealth.backend.dto.ChangePasswordDTO;
import com.mentalhealth.backend.dto.UpdateProfileDTO;
import com.mentalhealth.backend.vo.UserProfileVO;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

/**
 * 用户服务接口：覆盖认证（已有）与个人资料（新增）两类能力。
 * 所有涉及「当前用户」的方法均要求显式传入 userId，由 Controller 从 SecurityContext 解析。
 */
public interface UserService {

    /* ========== 认证相关（沿用原有实现） ========== */

    void register(String phone, String password, String code, String nickname);

    Map<String, Object> login(String phone, String password);

    void resetPassword(String phone, String password, String code);

    /* ========== 用户资料相关 ========== */

    /** 查询个人资料。 */
    UserProfileVO getProfile(Long userId);

    /** 更新昵称，返回更新后的完整资料。 */
    UserProfileVO updateProfile(Long userId, UpdateProfileDTO dto);

    /** 上传头像文件，落盘并回写 user.avatar，返回可访问 URL。 */
    String updateAvatar(Long userId, MultipartFile file);

    /** 校验旧密码后，用 BCrypt 加密新密码写回。 */
    void changePassword(Long userId, ChangePasswordDTO dto);
}
