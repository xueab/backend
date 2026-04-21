package com.mentalhealth.backend.mapper;

import com.mentalhealth.backend.entity.ChatSession;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface ChatSessionMapper {

    @Insert("INSERT INTO chat_session(user_id, title, status) VALUES(#{userId}, #{title}, #{status})")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int insert(ChatSession session);

    @Select("SELECT * FROM chat_session WHERE user_id = #{userId} ORDER BY created_at DESC")
    List<ChatSession> findByUserId(Long userId);

    @Select("SELECT * FROM chat_session WHERE id = #{id}")
    ChatSession findById(Long id);

    @Update("UPDATE chat_session SET status = #{status} WHERE id = #{id}")
    int updateStatus(@Param("id") Long id, @Param("status") Integer status);

    @Update("UPDATE chat_session SET report_url = #{reportUrl} WHERE id = #{id}")
    int updateReportUrl(@Param("id") Long id, @Param("reportUrl") String reportUrl);
}
