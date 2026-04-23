package com.mentalhealth.backend.mapper;

import com.mentalhealth.backend.entity.User;
import org.apache.ibatis.annotations.*;

@Mapper
public interface UserMapper {

    @Insert("INSERT INTO sys_user(username, password, nickname) VALUES(#{phone}, #{password}, #{nickname})")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int insert(User user);

    @Select("SELECT id, username AS phone, password, nickname, avatar, created_at, updated_at FROM sys_user WHERE username = #{phone}")
    User findByPhone(String phone);

    @Select("SELECT id, username AS phone, password, nickname, avatar, created_at, updated_at FROM sys_user WHERE id = #{id}")
    User findById(Long id);

    @Update("UPDATE sys_user SET password = #{password} WHERE username = #{phone}")
    int updatePassword(@Param("phone") String phone, @Param("password") String password);

    @Update("UPDATE sys_user SET password = #{password} WHERE id = #{id}")
    int updatePasswordById(@Param("id") Long id, @Param("password") String password);

    @Update("UPDATE sys_user SET nickname = #{nickname} WHERE id = #{id}")
    int updateNickname(@Param("id") Long id, @Param("nickname") String nickname);

    @Update("UPDATE sys_user SET avatar = #{avatar} WHERE id = #{id}")
    int updateAvatar(@Param("id") Long id, @Param("avatar") String avatar);
}
