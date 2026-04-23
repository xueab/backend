package com.mentalhealth.backend.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.net.Proxy;
import java.net.ProxySelector;
import java.net.URI;
import java.net.http.HttpClient;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.Executor;

@Configuration
public class AiStreamClientConfig {

    @Bean
    public HttpClient aiStreamHttpClient(AiServiceProperties properties) {
        // ai-service 使用 uvicorn(h11/httptools)，不支持 HTTP/2 的 h2c 升级，
        // 这里强制使用 HTTP/1.1，避免触发 "Unsupported upgrade request" 告警导致请求失败。
        return HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofMillis(properties.getConnectTimeoutMs()))
                .proxy(new NoProxySelector())
                .build();
    }

    @Bean(name = "chatSseExecutor")
    public Executor chatSseExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setThreadNamePrefix("chat-sse-");
        executor.setCorePoolSize(4);
        executor.setMaxPoolSize(8);
        executor.setQueueCapacity(100);
        executor.initialize();
        return executor;
    }

    /**
     * ai-service 为本机/内网服务，聊天流式调用强制直连，避免被系统代理劫持导致 502。
     */
    static class NoProxySelector extends ProxySelector {
        @Override
        public List<Proxy> select(URI uri) {
            return List.of(Proxy.NO_PROXY);
        }

        @Override
        public void connectFailed(URI uri, java.net.SocketAddress sa, java.io.IOException ioe) {
            // no-op
        }
    }
}
