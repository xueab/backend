package com.mentalhealth.backend.service.impl;

import com.mentalhealth.backend.common.BizException;
import com.mentalhealth.backend.config.JwtUtils;
import com.mentalhealth.backend.dto.ChangePasswordDTO;
import com.mentalhealth.backend.dto.UpdateProfileDTO;
import com.mentalhealth.backend.entity.User;
import com.mentalhealth.backend.mapper.UserMapper;
import com.mentalhealth.backend.service.CaptchaService;
import com.mentalhealth.backend.service.UserService;
import com.mentalhealth.backend.vo.UserProfileVO;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class UserServiceImpl implements UserService {

    private final UserMapper userMapper;
    private final CaptchaService captchaService;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtils jwtUtils;

    /** 头像本地存储根目录，默认 ./uploads。 */
    @Value("${upload.path:./uploads}")
    private String uploadPath;

    /** 头像可访问 URL 的前缀，默认 /uploads。 */
    @Value("${upload.base-url:/uploads}")
    private String uploadBaseUrl;

    /** 头像最大字节数，默认 5MB。 */
    @Value("${upload.avatar.max-size:5242880}")
    private long avatarMaxSize;

    private static final Set<String> ALLOWED_AVATAR_EXT = Set.of("jpg", "jpeg", "png", "gif", "webp");
    private static final Set<String> ALLOWED_AVATAR_MIME =
            Set.of("image/jpeg", "image/png", "image/gif", "image/webp");

    /* ========== 认证 ========== */

    @Override
    public void register(String phone, String password, String code, String nickname) {
        if (!captchaService.verify(phone, code)) {
            throw new BizException("验证码错误或已过期");
        }
        if (userMapper.findByPhone(phone) != null) {
            throw new BizException("该手机号已注册");
        }
        User user = new User();
        user.setPhone(phone);
        user.setPassword(passwordEncoder.encode(password));
        user.setNickname(nickname != null ? nickname : "用户" + phone.substring(phone.length() - 4));
        userMapper.insert(user);
    }

    @Override
    public Map<String, Object> login(String phone, String password) {
        User user = userMapper.findByPhone(phone);
        if (user == null) {
            throw new BizException("用户不存在");
        }
        if (!passwordEncoder.matches(password, user.getPassword())) {
            throw new BizException("账号或密码错误");
        }
        String token = jwtUtils.generateToken(user.getId(), user.getPhone());
        Map<String, Object> result = new HashMap<>();
        result.put("token", token);
        result.put("nickname", user.getNickname());
        result.put("avatar", user.getAvatar());
        result.put("userId", user.getId());
        return result;
    }

    @Override
    public void resetPassword(String phone, String password, String code) {
        if (!captchaService.verify(phone, code)) {
            throw new BizException("验证码错误或已过期");
        }
        User user = userMapper.findByPhone(phone);
        if (user == null) {
            throw new BizException("用户不存在");
        }
        userMapper.updatePassword(phone, passwordEncoder.encode(password));
    }

    /* ========== 用户资料 ========== */

    @Override
    public UserProfileVO getProfile(Long userId) {
        User user = userMapper.findById(userId);
        if (user == null) {
            throw new BizException("用户不存在");
        }
        return UserProfileVO.fromEntity(user);
    }

    @Override
    public UserProfileVO updateProfile(Long userId, UpdateProfileDTO dto) {
        if (dto == null || !StringUtils.hasText(dto.getNickname())) {
            throw new BizException("昵称不能为空");
        }
        String nickname = dto.getNickname().trim();
        if (nickname.length() > 32) {
            throw new BizException("昵称长度不能超过 32 个字符");
        }
        if (userMapper.findById(userId) == null) {
            throw new BizException("用户不存在");
        }
        userMapper.updateNickname(userId, nickname);
        return UserProfileVO.fromEntity(userMapper.findById(userId));
    }

    @Override
    public String updateAvatar(Long userId, MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new BizException("请选择要上传的头像文件");
        }
        if (file.getSize() > avatarMaxSize) {
            throw new BizException("头像大小不能超过 " + (avatarMaxSize / 1024 / 1024) + "MB");
        }
        String contentType = file.getContentType();
        if (contentType == null || !ALLOWED_AVATAR_MIME.contains(contentType.toLowerCase())) {
            throw new BizException("仅支持 jpg / png / gif / webp 格式");
        }
        String ext = resolveExtension(file.getOriginalFilename(), contentType);
        if (!ALLOWED_AVATAR_EXT.contains(ext)) {
            throw new BizException("仅支持 jpg / png / gif / webp 格式");
        }
        if (userMapper.findById(userId) == null) {
            throw new BizException("用户不存在");
        }

        String filename = UUID.randomUUID().toString().replace("-", "") + "." + ext;
        Path dir = Paths.get(uploadPath, "avatar").toAbsolutePath().normalize();
        try {
            Files.createDirectories(dir);
            Path target = dir.resolve(filename);
            file.transferTo(target.toFile());
        } catch (IOException e) {
            throw new BizException("头像保存失败：" + e.getMessage());
        }

        String baseUrl = uploadBaseUrl.endsWith("/")
                ? uploadBaseUrl.substring(0, uploadBaseUrl.length() - 1)
                : uploadBaseUrl;
        String accessUrl = baseUrl + "/avatar/" + filename;
        userMapper.updateAvatar(userId, accessUrl);
        return accessUrl;
    }

    @Override
    public void changePassword(Long userId, ChangePasswordDTO dto) {
        if (dto == null
                || !StringUtils.hasText(dto.getOldPassword())
                || !StringUtils.hasText(dto.getNewPassword())) {
            throw new BizException("旧密码与新密码均不能为空");
        }
        if (dto.getNewPassword().length() < 6 || dto.getNewPassword().length() > 32) {
            throw new BizException("新密码长度需在 6-32 位之间");
        }
        if (dto.getOldPassword().equals(dto.getNewPassword())) {
            throw new BizException("新密码不能与旧密码相同");
        }
        User user = userMapper.findById(userId);
        if (user == null) {
            throw new BizException("用户不存在");
        }
        if (!passwordEncoder.matches(dto.getOldPassword(), user.getPassword())) {
            throw new BizException("旧密码错误");
        }
        userMapper.updatePasswordById(userId, passwordEncoder.encode(dto.getNewPassword()));
    }

    /* ========== 私有工具 ========== */

    private String resolveExtension(String originalName, String contentType) {
        String ext = "";
        if (StringUtils.hasText(originalName) && originalName.contains(".")) {
            ext = originalName.substring(originalName.lastIndexOf('.') + 1).toLowerCase();
        }
        if (!ALLOWED_AVATAR_EXT.contains(ext)) {
            ext = switch (contentType.toLowerCase()) {
                case "image/jpeg" -> "jpg";
                case "image/png" -> "png";
                case "image/gif" -> "gif";
                case "image/webp" -> "webp";
                default -> "";
            };
        }
        return ext;
    }
}
