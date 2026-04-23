package com.mentalhealth.backend.service;

import com.mentalhealth.backend.common.BizException;
import com.mentalhealth.backend.config.JwtUtils;
import com.mentalhealth.backend.dto.ChangePasswordDTO;
import com.mentalhealth.backend.dto.UpdateProfileDTO;
import com.mentalhealth.backend.entity.User;
import com.mentalhealth.backend.mapper.UserMapper;
import com.mentalhealth.backend.service.impl.UserServiceImpl;
import com.mentalhealth.backend.vo.UserProfileVO;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.LocalDateTime;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * UserServiceImpl 单元测试示例：
 * 不拉起 Spring 容器，依赖通过 Mockito 打桩，验证关键业务分支。
 */
@ExtendWith(MockitoExtension.class)
class UserServiceImplTest {

    @Mock private UserMapper userMapper;
    @Mock private CaptchaService captchaService;
    @Mock private PasswordEncoder passwordEncoder;
    @Mock private JwtUtils jwtUtils;

    @InjectMocks private UserServiceImpl userService;

    @BeforeEach
    void setUp() {
        ReflectionTestUtils.setField(userService, "uploadPath", "./build/test-uploads");
        ReflectionTestUtils.setField(userService, "uploadBaseUrl", "/uploads");
        ReflectionTestUtils.setField(userService, "avatarMaxSize", 5L * 1024 * 1024);
    }

    private User sampleUser() {
        User u = new User();
        u.setId(1L);
        u.setPhone("13800000000");
        u.setPassword("encoded-old");
        u.setNickname("老昵称");
        u.setAvatar(null);
        u.setCreatedAt(LocalDateTime.now());
        return u;
    }

    /* ---------- getProfile ---------- */

    @Test
    void getProfile_success() {
        when(userMapper.findById(1L)).thenReturn(sampleUser());

        UserProfileVO vo = userService.getProfile(1L);

        assertNotNull(vo);
        assertEquals(1L, vo.getUserId());
        assertEquals("13800000000", vo.getPhone());
        assertEquals("老昵称", vo.getNickname());
    }

    @Test
    void getProfile_userNotFound() {
        when(userMapper.findById(999L)).thenReturn(null);
        BizException ex = assertThrows(BizException.class, () -> userService.getProfile(999L));
        assertEquals(400, ex.getCode());
    }

    /* ---------- updateProfile ---------- */

    @Test
    void updateProfile_success() {
        User before = sampleUser();
        User after = sampleUser();
        after.setNickname("新昵称");
        when(userMapper.findById(1L)).thenReturn(before, after);
        when(userMapper.updateNickname(1L, "新昵称")).thenReturn(1);

        UpdateProfileDTO dto = new UpdateProfileDTO();
        dto.setNickname("新昵称");

        UserProfileVO vo = userService.updateProfile(1L, dto);

        assertEquals("新昵称", vo.getNickname());
        verify(userMapper).updateNickname(1L, "新昵称");
    }

    @Test
    void updateProfile_blankNickname_throws() {
        UpdateProfileDTO dto = new UpdateProfileDTO();
        dto.setNickname("   ");
        assertThrows(BizException.class, () -> userService.updateProfile(1L, dto));
        verify(userMapper, never()).updateNickname(anyLong(), anyString());
    }

    /* ---------- changePassword ---------- */

    @Test
    void changePassword_success() {
        when(userMapper.findById(1L)).thenReturn(sampleUser());
        when(passwordEncoder.matches("oldPwd", "encoded-old")).thenReturn(true);
        when(passwordEncoder.encode("newPwd123")).thenReturn("encoded-new");

        ChangePasswordDTO dto = new ChangePasswordDTO();
        dto.setOldPassword("oldPwd");
        dto.setNewPassword("newPwd123");

        userService.changePassword(1L, dto);

        verify(userMapper).updatePasswordById(1L, "encoded-new");
    }

    @Test
    void changePassword_wrongOldPassword_throws() {
        when(userMapper.findById(1L)).thenReturn(sampleUser());
        when(passwordEncoder.matches("wrong", "encoded-old")).thenReturn(false);

        ChangePasswordDTO dto = new ChangePasswordDTO();
        dto.setOldPassword("wrong");
        dto.setNewPassword("newPwd123");

        BizException ex = assertThrows(BizException.class, () -> userService.changePassword(1L, dto));
        assertTrue(ex.getMessage().contains("旧密码"));
        verify(userMapper, never()).updatePasswordById(anyLong(), anyString());
    }

    @Test
    void changePassword_sameAsOld_throws() {
        ChangePasswordDTO dto = new ChangePasswordDTO();
        dto.setOldPassword("samePwd");
        dto.setNewPassword("samePwd");
        assertThrows(BizException.class, () -> userService.changePassword(1L, dto));
    }

    /* ---------- updateAvatar ---------- */

    @Test
    void updateAvatar_emptyFile_throws() {
        MockMultipartFile empty = new MockMultipartFile("file", "a.png", "image/png", new byte[0]);
        assertThrows(BizException.class, () -> userService.updateAvatar(1L, empty));
    }

    @Test
    void updateAvatar_unsupportedType_throws() {
        MockMultipartFile bad = new MockMultipartFile(
                "file", "a.txt", "text/plain", "hello".getBytes());
        assertThrows(BizException.class, () -> userService.updateAvatar(1L, bad));
    }

    @Test
    void updateAvatar_success_returnsUrlAndWritesBack() {
        when(userMapper.findById(1L)).thenReturn(sampleUser());
        MockMultipartFile png = new MockMultipartFile(
                "file", "me.png", "image/png", new byte[]{1, 2, 3, 4});

        String url = userService.updateAvatar(1L, png);

        assertNotNull(url);
        assertTrue(url.startsWith("/uploads/avatar/"));
        assertTrue(url.endsWith(".png"));
        verify(userMapper).updateAvatar(eq(1L), eq(url));
    }
}
