package com.mentalhealth.backend.common;

/**
 * 业务异常：由 Service 层主动抛出，统一由 GlobalExceptionHandler 转换为
 * { code: 400, msg, data: null } 的标准响应。
 */
public class BizException extends RuntimeException {

    private final int code;

    public BizException(String msg) {
        super(msg);
        this.code = 400;
    }

    public BizException(int code, String msg) {
        super(msg);
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
