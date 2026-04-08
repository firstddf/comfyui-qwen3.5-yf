# llama-server 使用指南

## 快速开始

### 1. 启动 llama-server

使用以下命令启动 llama-server:

```bash
# 基本启动命令
llama-server.exe -m models/LLM/your_model.gguf --mmproj models/LLM/your_mmproj.gguf
```

例如，使用 Qwen3.5:

```bash
llama-server.exe -m models/LLM/Qwen3.5/qwen3.5-2b-q8.gguf --mmproj models/LLM/Qwen3.5/qwen3.5-2b-mmproj-f16.gguf
```

### 2. 验证服务器运行

打开浏览器访问：http://127.0.0.1:8080

或者使用测试脚本:

```bash
python test_llama_server.py
```

### 3. 在 ComfyUI 中使用

在 llama-yf 节点中:

1. **启用 API 模式**: 设置 `use_api = True`
2. **配置 API 地址**: `api_url = http://127.0.0.1:8080`
3. **模型名称**: 留空 (自动检测) 或手动指定

## 模型名称说明

llama-server 会自动加载模型，并使用**模型文件名**作为模型 ID。

例如:
- 模型文件：`qwen3.5-2b-q8.gguf`
- 模型 ID: `qwen3.5-2b-q8` (不带.gguf 后缀)

### 查看可用模型

启动服务器后，访问：http://127.0.0.1:8080/v1/models

返回示例:

```json
{
  "data": [
    {"id": "qwen3.5-2b-q8"},
    {"id": "global"}
  ]
}
```

### 在节点中配置模型

**推荐方式**: 留空 `api_model` 参数，让节点自动检测服务器上的第一个模型。

**手动指定**: 如果服务器加载了多个模型，可以手动指定模型名称。

## 高级配置

### 多 GPU 支持

```bash
llama-server.exe -m models/LLM/model.gguf --mmproj models/LLM/mmproj.gguf -ngl 99
```

### 调整上下文大小

```bash
llama-server.exe -m models/LLM/model.gguf --mmproj models/LLM/mmproj.gguf -c 8192
```

### 绑定到特定端口

```bash
llama-server.exe -m models/LLM/model.gguf --mmproj models/LLM/mmproj.gguf --port 8081
```

然后在节点中设置 `api_url = http://127.0.0.1:8081`

## 常见问题

### Q: 节点报错 "model not found"
A: 检查以下几点:
1. 确保 llama-server 已启动
2. 访问 http://127.0.0.1:8080/v1/models 查看可用模型名称
3. 将模型名称填入节点的 `api_model` 参数，或留空使用自动检测

### Q: API 调用超时
A: 可能原因:
1. 服务器未启动或已停止
2. 防火墙阻止了 8080 端口
3. 模型加载时间过长

### Q: 图像推理失败
A: 确认:
1. 模型支持多模态 (有对应的 mmproj 文件)
2. llama-server 启动时指定了 `--mmproj` 参数
3. 图像尺寸不超过 `max_size` 限制

## 与 CLI 模式的区别

| 特性 | CLI 模式 (llama-mtmd-cli) | API 模式 (llama-server) |
|------|--------------------------|------------------------|
| 启动方式 | 每次推理时启动 | 常驻服务器 |
| 性能 | 每次冷启动 | 热启动，更快 |
| 显存占用 | 临时占用 | 持续占用 |
| 长 prompt 支持 | 有限制 | 更好支持 |
| 推荐场景 | 偶尔使用 | 频繁推理 |

## 测试工具

使用 `test_llama_server.py` 测试 API 连接:

```bash
python test_llama_server.py
```

测试内容:
1. ✅ 健康检查
2. ✅ 获取模型列表
3. ✅ 文本推理测试
4. ✅ 图像推理测试

## 示例工作流

### 1. 启动服务器

```bash
# 在后台启动 llama-server
start llama-server.exe -m models/LLM/qwen3.5-2b-q8.gguf --mmproj models/LLM/qwen3.5-2b-mmproj-f16.gguf --port 8080
```

### 2. 配置 ComfyUI 节点

- `use_api`: True
- `api_url`: http://127.0.0.1:8080
- `api_model`: (留空，自动检测)
- `inference_mode`: one by one / images / video
- 其他参数根据需要调整

### 3. 运行工作流

连接图像输入，点击 "Queue Prompt" 即可。

## 故障排除

### 查看服务器日志

llama-server 会在启动时输出详细信息，包括:
- 模型加载进度
- 可用层数
- 内存占用
- 错误信息

### 重启服务器

如果遇到问题，尝试:
1. 关闭现有服务器 (Ctrl+C)
2. 重新启动 llama-server
3. 等待健康检查通过 (访问 http://127.0.0.1:8080/health)

### 检查版本兼容性

确保 llama-server 版本与模型格式兼容。建议使用最新版本的 llama.cpp。
