package com.mentalhealth.backend.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.nio.file.Paths;

/**
 * 暴露本地 uploadPath 目录为静态资源，使前端可直接通过
 * {upload.base-url}/avatar/xxx.jpg 访问已上传的头像文件。
 */
@Configuration
public class WebMvcConfig implements WebMvcConfigurer {

    @Value("${upload.path:./uploads}")
    private String uploadPath;

    @Value("${upload.base-url:/uploads}")
    private String uploadBaseUrl;

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        String pattern = (uploadBaseUrl.endsWith("/") ? uploadBaseUrl : uploadBaseUrl + "/") + "**";
        String location = Paths.get(uploadPath).toAbsolutePath().normalize().toUri().toString();
        registry.addResourceHandler(pattern).addResourceLocations(location);
    }
}
