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

    @Update("UPDATE sys_user SET password = #{password} WHERE username = #{phone}")
    int updatePassword(@Param("phone") String phone, @Param("password") String password);
}
