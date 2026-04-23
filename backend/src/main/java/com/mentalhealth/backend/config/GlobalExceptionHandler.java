package com.mentalhealth.backend.config;

import com.mentalhealth.backend.common.BizException;
import com.mentalhealth.backend.common.Result;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

@RestControllerAdvice
public class GlobalExceptionHandler {

    /** 业务异常：按业务语义给前端 { code, msg, data:null }，默认 400。 */
    @ExceptionHandler(BizException.class)
    public Result<?> handleBizException(BizException e) {
        return Result.error(e.getCode(), e.getMessage());
    }

    /** 文件超过 multipart 限制（比 Service 层自检更靠前）。 */
    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public Result<?> handleUploadTooLarge(MaxUploadSizeExceededException e) {
        return Result.error(400, "上传文件过大");
    }

    @ExceptionHandler(RuntimeException.class)
    public Result<?> handleRuntimeException(RuntimeException e) {
        return Result.error(400, e.getMessage());
    }

    @ExceptionHandler(Exception.class)
    public Result<?> handleException(Exception e) {
        return Result.error(500, "服务器内部错误: " + e.getMessage());
    }
}
