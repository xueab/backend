package com.mentalhealth.backend.mapper;

import com.mentalhealth.backend.entity.MoodDiary;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface MoodDiaryMapper {

    @Insert("INSERT INTO mood_diary(user_id, content, mood_score, tags, ai_analysis) " +
            "VALUES(#{userId}, #{content}, #{moodScore}, #{tags}, #{aiAnalysis})")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int insert(MoodDiary diary);

    @Select("SELECT * FROM mood_diary WHERE user_id = #{userId} ORDER BY created_at DESC")
    List<MoodDiary> findByUserId(Long userId);

    @Select("SELECT * FROM mood_diary WHERE id = #{id}")
    MoodDiary findById(Long id);
}
