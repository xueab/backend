package com.mentalhealth.backend.service;

import com.mentalhealth.backend.config.JwtUtils;
import com.mentalhealth.backend.entity.User;
import com.mentalhealth.backend.mapper.UserMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class UserService {

    private final UserMapper userMapper;
    private final CaptchaService captchaService;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtils jwtUtils;

    public void register(String phone, String password, String code, String nickname) {
        if (!captchaService.verify(phone, code)) {
            throw new RuntimeException("验证码错误或已过期");
        }
        if (userMapper.findByPhone(phone) != null) {
            throw new RuntimeException("该手机号已注册");
        }
        User user = new User();
        user.setPhone(phone);
        user.setPassword(passwordEncoder.encode(password));
        user.setNickname(nickname != null ? nickname : "用户" + phone.substring(phone.length() - 4));
        userMapper.insert(user);
    }

    public Map<String, Object> login(String phone, String password) {
        User user = userMapper.findByPhone(phone);
        if (user == null) {
            throw new RuntimeException("用户不存在");
        }
        if (!passwordEncoder.matches(password, user.getPassword())) {
            throw new RuntimeException("账号或密码错误");
        }
        String token = jwtUtils.generateToken(user.getId(), user.getPhone());
        Map<String, Object> result = new HashMap<>();
        result.put("token", token);
        result.put("nickname", user.getNickname());
        result.put("avatar", user.getAvatar());
        result.put("userId", user.getId());
        return result;
    }

    public void resetPassword(String phone, String password, String code) {
        if (!captchaService.verify(phone, code)) {
            throw new RuntimeException("验证码错误或已过期");
        }
        User user = userMapper.findByPhone(phone);
        if (user == null) {
            throw new RuntimeException("用户不存在");
        }
        userMapper.updatePassword(phone, passwordEncoder.encode(password));
    }
}
