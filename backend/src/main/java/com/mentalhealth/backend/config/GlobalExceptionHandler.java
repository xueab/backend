package com.mentalhealth.backend.config;

import com.mentalhealth.backend.common.BizException;
import com.mentalhealth.backend.common.Result;
import jakarta.validation.ConstraintViolationException;
import org.springframework.validation.BindException;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

import java.util.stream.Collectors;

@RestControllerAdvice
public class GlobalExceptionHandler {

    /** 业务异常：按业务语义给前端 { code, msg, data:null }，默认 400。 */
    @ExceptionHandler(BizException.class)
    public Result<?> handleBizException(BizException e) {
        return Result.error(e.getCode(), e.getMessage());
    }

    /** @Valid / @RequestBody 参数校验失败。 */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public Result<?> handleValidation(MethodArgumentNotValidException e) {
        String msg = e.getBindingResult().getFieldErrors().stream()
                .map(FieldError::getDefaultMessage)
                .collect(Collectors.joining("; "));
        return Result.error(400, msg.isEmpty() ? "参数校验失败" : msg);
    }

    /** 表单绑定校验失败（如 GET 查询参数 @Valid）。 */
    @ExceptionHandler(BindException.class)
    public Result<?> handleBind(BindException e) {
        String msg = e.getBindingResult().getFieldErrors().stream()
                .map(FieldError::getDefaultMessage)
                .collect(Collectors.joining("; "));
        return Result.error(400, msg.isEmpty() ? "参数校验失败" : msg);
    }

    /** 方法参数上的 @NotNull/@Min 等约束失败。 */
    @ExceptionHandler(ConstraintViolationException.class)
    public Result<?> handleConstraint(ConstraintViolationException e) {
        String msg = e.getConstraintViolations().stream()
                .map(v -> v.getMessage())
                .collect(Collectors.joining("; "));
        return Result.error(400, msg.isEmpty() ? "参数校验失败" : msg);
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
