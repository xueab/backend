package com.mentalhealth.backend.service;

import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;

//生成、存储并校验手机验证码（短信验证码逻辑）
@Service
public class CaptchaService {

    /** phone -> {code, expireTime} */
    private final Map<String, long[]> store = new ConcurrentHashMap<>();

    private static final long VALID_MILLIS = 5 * 60 * 1000L;

    //TODO 改为真实发送验证码
    //TODO 改为redis存储验证码
    public String sendCode(String phone) {
        String code = String.format("%06d", ThreadLocalRandom.current().nextInt(1_000_000));
        store.put(phone, new long[]{Long.parseLong(code), System.currentTimeMillis() + VALID_MILLIS});
        System.out.println("【模拟短信】手机号: " + phone + ", 验证码: " + code + " (5分钟有效)");
        return code;
    }

    public boolean verify(String phone, String code) {
        long[] entry = store.get(phone);
        if (entry == null) {
            return false;
        }
        if (System.currentTimeMillis() > entry[1]) {
            store.remove(phone);
            return false;
        }
        boolean matched = String.valueOf(entry[0]).equals(code)
                || String.format("%06d", entry[0]).equals(code);
        if (matched) {
            store.remove(phone);
        }
        return matched;
    }
}
