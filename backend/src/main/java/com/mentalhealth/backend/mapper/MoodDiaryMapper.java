package com.mentalhealth.backend.mapper;

import com.mentalhealth.backend.entity.MoodDiary;
import org.apache.ibatis.annotations.*;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 情绪日记 Mapper。
 * 分页与条件过滤使用 MyBatis 的 &lt;script&gt; 动态 SQL 片段（等价于 MP 的 QueryWrapper），
 * 避免引入 MyBatis-Plus 带来与现有 starter 的冲突。
 */
@Mapper
public interface MoodDiaryMapper {

    @Insert("INSERT INTO mood_diary(user_id, content, mood_score, tags, ai_analysis) " +
            "VALUES(#{userId}, #{content}, #{moodScore}, #{tags}, #{aiAnalysis})")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int insert(MoodDiary diary);

    @Select("SELECT * FROM mood_diary WHERE user_id = #{userId} ORDER BY created_at DESC")
    List<MoodDiary> findByUserId(Long userId);

    /**
     * 按用户 + 时间区间拉取全部记录（不分页），供统计类接口在 Service 层用 Stream 聚合。
     * 时间区间采用「左闭右开」：startDate &lt;= created_at &lt; endDateExclusive。
     */
    @Select("SELECT * FROM mood_diary " +
            "WHERE user_id = #{userId} " +
            "AND created_at >= #{startDate} " +
            "AND created_at < #{endDateExclusive} " +
            "ORDER BY created_at ASC")
    List<MoodDiary> findByUserIdAndRange(@Param("userId") Long userId,
                                         @Param("startDate") LocalDateTime startDate,
                                         @Param("endDateExclusive") LocalDateTime endDateExclusive);

    @Select("SELECT * FROM mood_diary WHERE id = #{id}")
    MoodDiary findById(Long id);

    @Update("UPDATE mood_diary SET content = #{content}, mood_score = #{moodScore}, tags = #{tags} " +
            "WHERE id = #{id}")
    int update(MoodDiary diary);

    @Update("UPDATE mood_diary SET ai_analysis = #{aiAnalysis} WHERE id = #{id}")
    int updateAiAnalysis(@Param("id") Long id, @Param("aiAnalysis") String aiAnalysis);

    @Delete("DELETE FROM mood_diary WHERE id = #{id}")
    int deleteById(Long id);

    /**
     * 当天日记条数。使用 DATE(created_at) = CURDATE() 做简单按日聚合，
     * 若业务需要夸时区可改为传入 start/end 时间窗。
     */
    @Select("SELECT COUNT(*) FROM mood_diary " +
            "WHERE user_id = #{userId} AND DATE(created_at) = CURDATE()")
    long countToday(Long userId);

    /**
     * 按用户 + 时间区间 + 分值区间 分页查询。
     * 时间区间采用「左闭右开」：startDate &lt;= created_at &lt; endDateExclusive。
     * ORDER BY created_at DESC，满足前端「最新在前」的展示需求。
     */
    @Select({"<script>",
            "SELECT * FROM mood_diary",
            "WHERE user_id = #{userId}",
            "<if test='startDate != null'> AND created_at &gt;= #{startDate} </if>",
            "<if test='endDateExclusive != null'> AND created_at &lt; #{endDateExclusive} </if>",
            "<if test='minScore != null'> AND mood_score &gt;= #{minScore} </if>",
            "<if test='maxScore != null'> AND mood_score &lt;= #{maxScore} </if>",
            "ORDER BY created_at DESC",
            "LIMIT #{offset}, #{size}",
            "</script>"})
    List<MoodDiary> pageQuery(@Param("userId") Long userId,
                              @Param("startDate") LocalDateTime startDate,
                              @Param("endDateExclusive") LocalDateTime endDateExclusive,
                              @Param("minScore") Integer minScore,
                              @Param("maxScore") Integer maxScore,
                              @Param("offset") long offset,
                              @Param("size") long size);

    @Select({"<script>",
            "SELECT COUNT(*) FROM mood_diary",
            "WHERE user_id = #{userId}",
            "<if test='startDate != null'> AND created_at &gt;= #{startDate} </if>",
            "<if test='endDateExclusive != null'> AND created_at &lt; #{endDateExclusive} </if>",
            "<if test='minScore != null'> AND mood_score &gt;= #{minScore} </if>",
            "<if test='maxScore != null'> AND mood_score &lt;= #{maxScore} </if>",
            "</script>"})
    long pageCount(@Param("userId") Long userId,
                   @Param("startDate") LocalDateTime startDate,
                   @Param("endDateExclusive") LocalDateTime endDateExclusive,
                   @Param("minScore") Integer minScore,
                   @Param("maxScore") Integer maxScore);
}
