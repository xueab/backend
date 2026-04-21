package com.mentalhealth.backend.controller;

import com.mentalhealth.backend.common.Result;
import com.mentalhealth.backend.dto.CodeRequest;
import com.mentalhealth.backend.dto.LoginRequest;
import com.mentalhealth.backend.dto.RegisterRequest;
import com.mentalhealth.backend.dto.ResetPasswordRequest;
import com.mentalhealth.backend.service.CaptchaService;
import com.mentalhealth.backend.service.UserService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final CaptchaService captchaService;
    private final UserService userService;

    @PostMapping("/code")
    public Result<?> sendCode(@RequestBody CodeRequest request) {
        captchaService.sendCode(request.getPhone());
        return Result.success("验证码已发送");
    }

    @PostMapping("/register")
    public Result<?> register(@RequestBody RegisterRequest request) {
        userService.register(request.getPhone(), request.getPassword(),
                request.getCode(), request.getNickname());
        return Result.success("注册成功");
    }

    @PostMapping("/login")
    public Result<Map<String, Object>> login(@RequestBody LoginRequest request) {
        Map<String, Object> data = userService.login(request.getPhone(), request.getPassword());
        return Result.success(data);
    }

    @PostMapping("/reset-password")
    public Result<?> resetPassword(@RequestBody ResetPasswordRequest request) {
        userService.resetPassword(request.getPhone(), request.getPassword(), request.getCode());
        return Result.success("密码重置成功");
    }
}
