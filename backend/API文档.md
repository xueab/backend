# 心理健康系统后端 API 文档

## 1. 文档说明

本文档基于当前 `backend` 项目现有代码整理，可直接作为前端接口设计和联调用文档。

- 当前后端**实际已开放**的接口只有认证模块，共 4 个接口
- 当前统一响应结构为 `Result<T>`
- 当前项目使用 `JWT Bearer Token` 鉴权
- 除 `/api/auth/**` 之外，其余接口默认都需要登录，但目前代码中还没有开放其他业务 Controller

## 2. 基础信息

### 2.1 Base URL

当前项目未单独配置 `server.port`，Spring Boot 默认端口为：

```text
http://localhost:8080
```

接口完整前缀示例：

```text
http://localhost:8080/api/auth/login
```

### 2.2 前端跨域

当前仅允许以下前端地址跨域访问：

```text
http://localhost:5173
```

### 2.3 鉴权方式

- 认证接口 `/api/auth/**`：无需携带 Token
- 其他接口：需要在请求头中携带 JWT

请求头格式：

```http
Authorization: Bearer <token>
```

## 3. 统一响应格式

后端统一返回：

```json
{
  "code": 200,
  "msg": "success",
  "data": {}
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | number | 业务状态码，成功通常为 `200` |
| `msg` | string | 响应消息，成功固定为 `success` |
| `data` | any | 实际返回数据，可能是对象、字符串或 `null` |

### 3.1 成功响应示例

#### 返回对象

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "token": "xxx",
    "nickname": "小明",
    "avatar": "https://example.com/avatar.png",
    "userId": 1
  }
}
```

#### 返回字符串

```json
{
  "code": 200,
  "msg": "success",
  "data": "注册成功"
}
```

### 3.2 失败响应示例

#### 业务异常

```json
{
  "code": 400,
  "msg": "验证码错误或已过期",
  "data": null
}
```

#### 系统异常

```json
{
  "code": 500,
  "msg": "服务器内部错误: xxx",
  "data": null
}
```

## 4. 接口总览

| 接口名称 | 方法 | 路径 | 是否鉴权 | 说明 |
| --- | --- | --- | --- | --- |
| 发送验证码 | `POST` | `/api/auth/code` | 否 | 发送手机验证码 |
| 用户注册 | `POST` | `/api/auth/register` | 否 | 手机号注册 |
| 用户登录 | `POST` | `/api/auth/login` | 否 | 登录并获取 Token |
| 重置密码 | `POST` | `/api/auth/reset-password` | 否 | 通过验证码重置密码 |

## 5. 详细接口文档

---

### 5.1 发送验证码

- 请求方法：`POST`
- 请求路径：`/api/auth/code`
- 是否需要登录：否
- Content-Type：`application/json`

#### 请求参数

```json
{
  "phone": "13800138000"
}
```

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `phone` | string | 是 | 手机号 |

#### 成功响应

```json
{
  "code": 200,
  "msg": "success",
  "data": "验证码已发送"
}
```

#### 说明

- 当前验证码为**模拟发送**
- 验证码不会通过接口返回给前端
- 验证码会打印在后端控制台日志中，便于本地开发测试
- 验证码有效期为 **5 分钟**
- 验证码校验成功后会立即失效，不能重复使用

---

### 5.2 用户注册

- 请求方法：`POST`
- 请求路径：`/api/auth/register`
- 是否需要登录：否
- Content-Type：`application/json`

#### 请求参数

```json
{
  "phone": "13800138000",
  "password": "123456",
  "code": "654321",
  "nickname": "小明"
}
```

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `phone` | string | 是 | 手机号，系统按唯一账号处理 |
| `password` | string | 是 | 登录密码，后端会加密存储 |
| `code` | string | 是 | 手机验证码 |
| `nickname` | string | 否 | 昵称；不传时后端默认生成 `用户xxxx` |

#### 成功响应

```json
{
  "code": 200,
  "msg": "success",
  "data": "注册成功"
}
```

#### 可能的失败响应

```json
{
  "code": 400,
  "msg": "验证码错误或已过期",
  "data": null
}
```

```json
{
  "code": 400,
  "msg": "该手机号已注册",
  "data": null
}
```

#### 说明

- 当前项目没有做表单注解校验，前端需要自行校验手机号、密码、验证码是否为空
- 昵称可以不传，不传时后端会自动生成默认昵称

---

### 5.3 用户登录

- 请求方法：`POST`
- 请求路径：`/api/auth/login`
- 是否需要登录：否
- Content-Type：`application/json`

#### 请求参数

```json
{
  "phone": "13800138000",
  "password": "123456"
}
```

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `phone` | string | 是 | 手机号 |
| `password` | string | 是 | 登录密码 |

#### 成功响应

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "token": "jwt-token",
    "nickname": "小明",
    "avatar": "https://example.com/avatar.png",
    "userId": 1
  }
}
```

#### `data` 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `token` | string | JWT 登录令牌 |
| `nickname` | string | 用户昵称 |
| `avatar` | string \| null | 用户头像地址，当前可能为空 |
| `userId` | number | 用户 ID |

#### 可能的失败响应

```json
{
  "code": 400,
  "msg": "用户不存在",
  "data": null
}
```

```json
{
  "code": 400,
  "msg": "账号或密码错误",
  "data": null
}
```

#### 说明

- Token 过期时间为 **86400 秒（24 小时）**
- 登录成功后，前端应保存 `token`
- 后续访问受保护接口时，在请求头中添加：

```http
Authorization: Bearer jwt-token
```

---

### 5.4 重置密码

- 请求方法：`POST`
- 请求路径：`/api/auth/reset-password`
- 是否需要登录：否
- Content-Type：`application/json`

#### 请求参数

```json
{
  "phone": "13800138000",
  "password": "newPassword123",
  "code": "654321"
}
```

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `phone` | string | 是 | 手机号 |
| `password` | string | 是 | 新密码 |
| `code` | string | 是 | 手机验证码 |

#### 成功响应

```json
{
  "code": 200,
  "msg": "success",
  "data": "密码重置成功"
}
```

#### 可能的失败响应

```json
{
  "code": 400,
  "msg": "验证码错误或已过期",
  "data": null
}
```

```json
{
  "code": 400,
  "msg": "用户不存在",
  "data": null
}
```

## 6. 前端建议的 TypeScript 类型

```ts
export interface ApiResponse<T = any> {
  code: number;
  msg: string;
  data: T;
}

export interface SendCodeRequest {
  phone: string;
}

export interface RegisterRequest {
  phone: string;
  password: string;
  code: string;
  nickname?: string;
}

export interface LoginRequest {
  phone: string;
  password: string;
}

export interface ResetPasswordRequest {
  phone: string;
  password: string;
  code: string;
}

export interface LoginResponseData {
  token: string;
  nickname: string;
  avatar: string | null;
  userId: number;
}
```

## 7. 当前已存在但未开放的业务模块

从当前代码结构看，后端已经有对应实体和数据库访问层，但**暂时没有 Controller，对前端还不能直接调用**：

### 7.1 用户信息

已有实体字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | number | 用户 ID |
| `phone` | string | 手机号 |
| `password` | string | 加密密码 |
| `nickname` | string | 昵称 |
| `avatar` | string \| null | 头像 |
| `createdAt` | string | 创建时间 |
| `updatedAt` | string | 更新时间 |

### 7.2 情绪日记

已有实体字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | number | 日记 ID |
| `userId` | number | 用户 ID |
| `content` | string | 日记内容 |
| `moodScore` | number | 情绪分值，1-10 |
| `tags` | string | 情绪标签，逗号分隔 |
| `aiAnalysis` | string | AI 分析结果 |
| `createdAt` | string | 创建时间 |

### 7.3 AI 对话会话

已有会话实体字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | number | 会话 ID |
| `userId` | number | 用户 ID |
| `title` | string | 会话标题 |
| `status` | number | `0` 进行中，`1` 已结束 |
| `reportUrl` | string \| null | 报告地址 |
| `createdAt` | string | 创建时间 |

已有消息实体字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | number | 消息 ID |
| `sessionId` | number | 会话 ID |
| `role` | string | `user` / `assistant` |
| `content` | string | 消息内容 |
| `createdAt` | string | 创建时间 |

## 8. 联调注意事项

1. 当前只有认证接口可直接联调，其他业务功能前端如果要先做页面，可先按第 7 节数据结构预留类型。
2. 验证码接口目前是模拟实现，联调时需要查看后端控制台中的验证码。
3. 当前后端未做严格参数校验，前端应补充必填校验、手机号格式校验、密码强度校验。
4. 登录成功后应统一封装 Token 请求头。
5. 成功时 `msg` 固定是 `success`，真正的提示文案通常在 `data` 里，例如 `"注册成功"`、`"密码重置成功"`。

## 9. 建议的前端接口模块划分

前端当前可按以下方式组织：

- `authApi`
  - `sendCode`
  - `register`
  - `login`
  - `resetPassword`
- `types/api`
  - 公共响应类型
  - 认证模块请求/响应类型
- `types/business`
  - 用户信息类型
  - 情绪日记类型
  - 聊天会话与消息类型

---

如后端后续新增 `用户信息 / 情绪日记 / AI 聊天` 的 Controller，可以在这份文档基础上继续补充。
