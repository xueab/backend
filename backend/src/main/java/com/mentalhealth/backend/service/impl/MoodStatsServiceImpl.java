package com.mentalhealth.backend.service.impl;

import com.mentalhealth.backend.common.BizException;
import com.mentalhealth.backend.entity.MoodDiary;
import com.mentalhealth.backend.mapper.MoodDiaryMapper;
import com.mentalhealth.backend.service.MoodStatsService;
import com.mentalhealth.backend.vo.MoodInsightVO;
import com.mentalhealth.backend.vo.MoodSummaryVO;
import com.mentalhealth.backend.vo.MoodTrendVO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@Service
@RequiredArgsConstructor
public class MoodStatsServiceImpl implements MoodStatsService {

    private static final Set<Integer> ALLOWED_RANGES = Set.copyOf(EnumSet.of(Range.R7, Range.R14, Range.R30)
            .stream().map(Range::getDays).collect(Collectors.toSet()));

    private static final DateTimeFormatter LABEL_FMT = DateTimeFormatter.ofPattern("MM-dd");

    private final MoodDiaryMapper moodDiaryMapper;

    @Override
    public MoodTrendVO trend(Long userId, Integer range) {
        int days = validateRange(range);
        LocalDate endDate = LocalDate.now();
        LocalDate startDate = endDate.minusDays(days - 1L);

        Map<LocalDate, Double> avgByDate = loadAndGroupAverage(userId, startDate, endDate);

        List<String> dates = new ArrayList<>(days);
        List<Double> scores = new ArrayList<>(days);
        IntStream.range(0, days).forEach(i -> {
            LocalDate d = startDate.plusDays(i);
            dates.add(d.format(LABEL_FMT));
            scores.add(avgByDate.get(d));
        });

        return MoodTrendVO.builder().dates(dates).scores(scores).build();
    }

    @Override
    public MoodSummaryVO summary(Long userId, Integer range) {
        int days = validateRange(range);
        LocalDate endDate = LocalDate.now();
        LocalDate startDate = endDate.minusDays(days - 1L);

        List<MoodDiary> diaries = loadDiaries(userId, startDate, endDate);
        if (diaries.isEmpty()) {
            return MoodSummaryVO.builder().average(0.0).max(0).min(0).days(0).build();
        }

        List<Integer> scoreList = diaries.stream()
                .map(MoodDiary::getMoodScore)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
        if (scoreList.isEmpty()) {
            return MoodSummaryVO.builder().average(0.0).max(0).min(0).days(0).build();
        }

        double average = round1(scoreList.stream().mapToInt(Integer::intValue).average().orElse(0));
        int max = scoreList.stream().mapToInt(Integer::intValue).max().orElse(0);
        int min = scoreList.stream().mapToInt(Integer::intValue).min().orElse(0);
        int daysWithRecord = (int) diaries.stream()
                .filter(d -> d.getCreatedAt() != null)
                .map(d -> d.getCreatedAt().toLocalDate())
                .distinct()
                .count();

        return MoodSummaryVO.builder()
                .average(average)
                .max(max)
                .min(min)
                .days(daysWithRecord)
                .build();
    }

    @Override
    public MoodInsightVO insights(Long userId, Integer range) {
        int days = validateRange(range);
        LocalDate endDate = LocalDate.now();
        LocalDate startDate = endDate.minusDays(days - 1L);

        List<MoodDiary> diaries = loadDiaries(userId, startDate, endDate);
        List<String> tips = new ArrayList<>();

        List<MoodDiary> valid = diaries.stream()
                .filter(d -> d.getMoodScore() != null && d.getCreatedAt() != null)
                .collect(Collectors.toList());

        if (valid.isEmpty()) {
            tips.add("最近 " + days + " 天还没有日记记录，试着写下第一篇，记录一下心情吧。");
            return MoodInsightVO.builder().tips(tips).build();
        }

        double average = valid.stream().mapToInt(MoodDiary::getMoodScore).average().orElse(0);
        double avgRounded = round1(average);

        if (avgRounded >= 7) {
            tips.add("最近整体情绪不错，平均分 " + avgRounded + "，继续保持这份好状态～");
        } else if (avgRounded >= 5) {
            tips.add("情绪平稳，平均分 " + avgRounded + "，可以尝试安排一些让自己开心的小事。");
        } else {
            tips.add("最近有点辛苦，平均分 " + avgRounded + "，记得好好休息，必要时和信任的人聊聊。");
        }

        MoodDiary happiest = valid.stream()
                .max((a, b) -> {
                    int c = Integer.compare(a.getMoodScore(), b.getMoodScore());
                    return c != 0 ? c : a.getCreatedAt().compareTo(b.getCreatedAt());
                })
                .orElseThrow();
        MoodDiary lowest = valid.stream()
                .min((a, b) -> {
                    int c = Integer.compare(a.getMoodScore(), b.getMoodScore());
                    return c != 0 ? c : a.getCreatedAt().compareTo(b.getCreatedAt());
                })
                .orElseThrow();

        tips.add("最开心的是 " + happiest.getCreatedAt().toLocalDate().format(LABEL_FMT)
                + "，当天心情 " + happiest.getMoodScore() + " 分。");
        tips.add("最低分是 " + lowest.getMoodScore() + " 分，出现在 "
                + lowest.getCreatedAt().toLocalDate().format(LABEL_FMT) + "。");

        return MoodInsightVO.builder().tips(tips).build();
    }

    /* ========== 私有工具 ========== */

    /**
     * 校验 range 仅允许 7 / 14 / 30，其他值抛 400。
     */
    private int validateRange(Integer range) {
        if (range == null || !ALLOWED_RANGES.contains(range)) {
            throw new BizException(400, "range 仅支持 7 / 14 / 30");
        }
        return range;
    }

    /**
     * 拉取 [startDate, endDate] 区间（含两端）的日记，按 created_at 升序。
     * 为了契合 mapper 的「左闭右开」约定，endExclusive 取 endDate 的次日 0 点。
     */
    private List<MoodDiary> loadDiaries(Long userId, LocalDate startDate, LocalDate endDate) {
        LocalDateTime start = startDate.atStartOfDay();
        LocalDateTime endExclusive = endDate.plusDays(1).atStartOfDay();
        List<MoodDiary> list = moodDiaryMapper.findByUserIdAndRange(userId, start, endExclusive);
        return list == null ? Collections.emptyList() : list;
    }

    /**
     * 按日期分组，计算每天的平均 moodScore（保留 1 位小数）。
     * 无记录的日期不会出现在 Map 中，由调用方自行补齐 null。
     */
    private Map<LocalDate, Double> loadAndGroupAverage(Long userId, LocalDate startDate, LocalDate endDate) {
        return loadDiaries(userId, startDate, endDate).stream()
                .filter(d -> d.getMoodScore() != null && d.getCreatedAt() != null)
                .collect(Collectors.groupingBy(
                        d -> d.getCreatedAt().toLocalDate(),
                        Collectors.collectingAndThen(
                                Collectors.averagingInt(MoodDiary::getMoodScore),
                                avg -> round1(avg == null ? 0.0 : avg))));
    }

    private static double round1(double v) {
        return Math.round(v * 10.0) / 10.0;
    }

    /** 仅用于收敛 ALLOWED_RANGES 常量，避免散落魔法数字。 */
    private enum Range {
        R7(7), R14(14), R30(30);

        private final int days;

        Range(int days) {
            this.days = days;
        }

        int getDays() {
            return days;
        }
    }
}
