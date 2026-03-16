# Gemini_Pipeline

Google Gemini Pipeline（用于 Open WebUI 的 Gemini 管道）。

## 功能概览

- 支持 Gemini 文本与多模态请求。
- 支持多 API Key 轮询，并在额度/限流错误时自动切换 Key。
- 模型列表改为手动配置（不再自动从 Google API 拉取）。
- 支持流式与非流式响应。
- 支持 Thinking（`THINKING_BUDGET` / `THINKING_LEVEL`）。
- 支持可选宽松安全策略。
- 支持图片智能压缩、缩放与历史图片复用。
- 支持「空回复重试」与「临时错误重试」分离控制。

## 快速配置

至少配置以下变量：

- `GOOGLE_API_KEY`
- `GOOGLE_MODEL_LIST`

示例：

```env
GOOGLE_API_KEY=your_api_key_1,your_api_key_2
GOOGLE_MODEL_LIST=gemini-2.5-flash,gemini-2.5-pro:Gemini 2.5 Pro
GOOGLE_API_VERSION=v1alpha
```

## 模型列表配置（手动）

### `GOOGLE_MODEL_LIST`

手动指定可用模型列表，逗号分隔，支持两种格式：

- `模型ID`
- `模型ID:显示名称`

示例：

```env
GOOGLE_MODEL_LIST=gemini-2.5-flash,gemini-2.5-pro:Gemini 2.5 Pro,gemini-2.0-flash
```

## 重试逻辑说明（重点）

### `GOOGLE_RETRY_COUNT`

控制 API 临时错误重试次数（例如服务端错误、限流/额度类可恢复问题）。

### `GOOGLE_EMPTY_RESPONSE_RETRY_COUNT`

控制“模型返回空内容”时的重试次数，**独立于** `GOOGLE_RETRY_COUNT`。

- `0`：关闭空回复重试
- `1/2/3/...`：按次数重试
- `-1`：兼容旧配置，回退为 `GOOGLE_RETRY_COUNT`（已不推荐）

默认值为 `2`。

## 主要环境变量

### 连接与基础

- `GOOGLE_GENAI_BASE_URL`：API 基础 URL。
- `GOOGLE_API_KEY`：API Key，支持多个（逗号分隔）。
- `GOOGLE_API_VERSION`：API 版本（如 `v1alpha` / `v1beta` / `v1`）。
- `GOOGLE_STREAMING_ENABLED`：是否启用流式输出。
- `GOOGLE_ENABLE_FORWARD_USER_INFO_HEADERS`：是否透传用户信息请求头。

### 思考与安全

- `GOOGLE_INCLUDE_THOUGHTS`：是否输出思考内容。
- `GOOGLE_THINKING_BUDGET`：Gemini 2.5 Thinking 预算。
- `GOOGLE_THINKING_LEVEL`：Gemini 3 Thinking 等级。
- `GOOGLE_USE_PERMISSIVE_SAFETY`：是否启用宽松安全策略。

### 模型

- `GOOGLE_MODEL_LIST`：手动模型列表。
- `GOOGLE_ENABLE_GEMINI_25_FLASH_SEARCH_MODEL`：是否额外增加 `gemini-2.5-flash-search`。

### 重试

- `GOOGLE_RETRY_COUNT`：API 临时错误重试次数。
- `GOOGLE_EMPTY_RESPONSE_RETRY_COUNT`：空回复重试次数（独立控制）。

### 图片处理

- `GOOGLE_IMAGE_ENABLE_OPTIMIZATION`：是否启用图片自动优化（压缩/缩放/格式转换），建议保持 `true`。
- `GOOGLE_IMAGE_MAX_SIZE_MB`：单图目标体积上限（MB），超出会触发压缩。
- `GOOGLE_IMAGE_MAX_DIMENSION`：单图最大边长（像素），超出会等比缩放。
- `GOOGLE_IMAGE_COMPRESSION_QUALITY`：JPEG 压缩质量基准值（1-100，越高越清晰但体积更大）。
- `GOOGLE_IMAGE_PNG_THRESHOLD_MB`：PNG 超过该体积时优先转 JPEG 以提升压缩率。
- `GOOGLE_IMAGE_HISTORY_MAX_REFERENCES`：一次请求最多携带的图片总数（历史 + 当前消息），用于控制上下文体积。
- `GOOGLE_IMAGE_ADD_LABELS`：是否在每张图片前加 `[Image 1]` 这类标签，便于模型指代。
- `GOOGLE_IMAGE_DEDUP_HISTORY`：是否对历史图片按内容去重，避免重复传同一张图。
- `GOOGLE_IMAGE_HISTORY_FIRST`：图片拼接顺序；`true` 先历史后当前，`false` 先当前后历史。

## 建议配置

```env
# 基础
GOOGLE_API_KEY=your_key
GOOGLE_MODEL_LIST=gemini-2.5-flash,gemini-2.5-pro

# 重试
GOOGLE_RETRY_COUNT=2
GOOGLE_EMPTY_RESPONSE_RETRY_COUNT=2

# 可选
GOOGLE_STREAMING_ENABLED=true
GOOGLE_INCLUDE_THOUGHTS=true
```
