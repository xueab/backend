package com.mentalhealth.backend.config;

import lombok.RequiredArgsConstructor;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.util.StringUtils;
import org.springframework.web.client.RestClient;

@Configuration
@RequiredArgsConstructor
@EnableConfigurationProperties(AiServiceProperties.class)
public class AiServiceClientConfig {

    @Bean
    public RestClient aiServiceRestClient(RestClient.Builder builder, AiServiceProperties properties) {
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(properties.getConnectTimeoutMs());
        requestFactory.setReadTimeout(properties.getReadTimeoutMs());

        RestClient.Builder restClientBuilder = builder
                .baseUrl(properties.getBaseUrl())
                .requestFactory(requestFactory);

        if (StringUtils.hasText(properties.getInternalToken())) {
            restClientBuilder.defaultHeader("X-Internal-Token", properties.getInternalToken());
        }

        return restClientBuilder.build();
    }
}
