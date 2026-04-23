package com.mentalhealth.backend.controller;

import com.mentalhealth.backend.common.Result;
import com.mentalhealth.backend.common.SecurityUtils;
import com.mentalhealth.backend.dto.ChangePasswordDTO;
import com.mentalhealth.backend.dto.UpdateProfileDTO;
import com.mentalhealth.backend.service.UserService;
import com.mentalhealth.backend.vo.UserProfileVO;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

/**
 * 用户资料模块控制器。
 * 路径前缀 /api/user，SecurityConfig 中 anyRequest().authenticated() 保证了必须携带 JWT；
 * Controller 本身不再接收 userId 参数，统一通过 SecurityUtils 从 SecurityContext 获取。
 */
@RestController
@RequestMapping("/api/user")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    @GetMapping("/profile")
    public Result<UserProfileVO> getProfile() {
        Long userId = SecurityUtils.getCurrentUserId();
        return Result.success(userService.getProfile(userId));
    }

    @PutMapping("/profile")
    public Result<UserProfileVO> updateProfile(@RequestBody UpdateProfileDTO dto) {
        Long userId = SecurityUtils.getCurrentUserId();
        return Result.success(userService.updateProfile(userId, dto));
    }

    @PostMapping(value = "/avatar", consumes = "multipart/form-data")
    public Result<Map<String, String>> uploadAvatar(@RequestPart("file") MultipartFile file) {
        Long userId = SecurityUtils.getCurrentUserId();
        String avatarUrl = userService.updateAvatar(userId, file);
        return Result.success(Map.of("avatar", avatarUrl));
    }

    @PutMapping("/password")
    public Result<?> changePassword(@RequestBody ChangePasswordDTO dto) {
        Long userId = SecurityUtils.getCurrentUserId();
        userService.changePassword(userId, dto);
        return Result.success("密码修改成功");
    }
}
