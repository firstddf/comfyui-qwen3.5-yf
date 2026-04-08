# llama-server 故障排除指南

## ✅ 验证服务器正常运行的步骤

### 1. 检查服务器是否启动

访问：http://127.0.0.1:8080/health

预期返回:
```json
{"status":"ok"}
```

### 2. 检查可用模型

访问：http://127.0.0.1:8080/v1/models

预期返回类似:
```json
{
  "data": [
    {"id": "qwen3.5-2b-q8"},
    {"id": "global"}
  ]
}
```

### 3. 运行测试脚本

```bash
# 基础 API 测试
python test_llama_server.py

# 节点调用模拟测试
python test_node_simulation.py
```

## ❌ 常见错误及解决方案

### 错误 1: "Connection refused" 或 "Server is not running"

**症状**: 节点报错无法连接到 API

**解决方案**:
1. 确保 llama-server 已启动
2. 检查端口是否正确 (默认 8080)
3. 检查防火墙设置

```bash
# 启动服务器
llama-server.exe -m models/LLM/your_model.gguf --mmproj models/LLM/your_mmproj.gguf
```

### 错误 2: "model 'xxx' not found" (400)

**症状**: API 返回 400 错误，提示模型未找到

**原因**: 节点中指定的 `api_model` 名称与服务器中的模型 ID 不匹配

**解决方案**:
1. 访问 http://127.0.0.1:8080/v1/models 查看实际模型 ID
2. 将 `api_model` 设置为正确的模型 ID
3. **推荐**: 留空 `api_model`,让节点自动检测

### 错误 3: 图像推理失败，但文本推理正常

**症状**: 纯文本可以工作，但带图像时报错

**原因**: 模型不支持多模态或 mmproj 文件未正确加载

**解决方案**:
1. 确认使用的是多模态模型 (如 Qwen-VL)
2. 启动时指定 mmproj 文件:
   ```bash
   llama-server.exe -m model.gguf --mmproj mmproj.gguf
   ```
3. 检查日志中是否显示 mmproj 加载成功

### 错误 4: 响应格式不正确

**症状**: 节点无法解析 API 响应

**检查响应格式**:
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"your-model","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":50}'
```

预期返回:
```json
{
  "choices": [{
    "message": {
      "content": "Hello!"
    }
  }]
}
```

### 错误 5: 超时或响应极慢

**症状**: 推理时间过长或超时

**解决方案**:
1. 减少 `max_tokens` 参数
2. 减少 `ctx_size` 参数
3. 使用更小的模型
4. 增加 GPU 层数 (`-ngl 99`)
5. 检查显存是否充足

## 🔧 调试技巧

### 查看详细日志

启动服务器时添加 `--verbose` 参数:
```bash
llama-server.exe -m model.gguf --mmproj mmproj.gguf --verbose
```

### 检查 API 请求/响应

使用测试脚本 `test_node_simulation.py` 查看完整的请求和响应:
```bash
python test_node_simulation.py
```

### 手动测试 API

使用 curl 或 Postman 测试 API:
```bash
# 文本推理
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "你好"}]}
    ],
    "max_tokens": 100
  }'

# 图像推理
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,YOUR_BASE64_IMAGE"}},
        {"type": "text", "text": "描述这张图片"}
      ]}
    ],
    "max_tokens": 100
  }'
```

## 📋 节点配置检查清单

在 ComfyUI 的 llama-yf 节点中:

- [ ] `use_api`: 设置为 `True`
- [ ] `api_url`: `http://127.0.0.1:8080` (确保与服务器端口一致)
- [ ] `api_model`: 留空 (推荐) 或填写正确的模型 ID
- [ ] `inference_mode`: 选择正确的模式 (one by one / images / video)
- [ ] 图像输入: 已连接到 `image` 接口
- [ ] 模型文件: 已在服务器中正确加载

## 🚀 推荐的服务器启动命令

### 基础配置
```bash
llama-server.exe -m models/LLM/qwen3.5-2b-q8.gguf --mmproj models/LLM/qwen3.5-2b-mmproj-f16.gguf
```

### GPU 加速 (推荐)
```bash
llama-server.exe -m models/LLM/qwen3.5-2b-q8.gguf --mmproj models/LLM/qwen3.5-2b-mmproj-f16.gguf -ngl 99
```

### 自定义端口
```bash
llama-server.exe -m models/LLM/qwen3.5-2b-q8.gguf --mmproj models/LLM/qwen3.5-2b-mmproj-f16.gguf --port 8081
```
然后在节点中设置 `api_url = http://127.0.0.1:8081`

### 大上下文窗口
```bash
llama-server.exe -m models/LLM/qwen3.5-2b-q8.gguf --mmproj models/LLM/qwen3.5-2b-mmproj-f16.gguf -c 8192 -ngl 99
```

## 💡 性能优化建议

1. **使用 GPU**: 添加 `-ngl 99` 参数加载所有层到 GPU
2. **常驻服务器**: llama-server 启动后保持运行，避免频繁重启
3. **合理设置 max_tokens**: 不要设置过大 (默认 4096 足够)
4. **批量处理**: 使用 `images` 模式一次处理多张图像
5. **监控显存**: 使用 GPU-Z 或任务管理器监控显存使用

## 📞 获取帮助

如果以上方法都无法解决问题:

1. 检查 llama-server 启动日志
2. 运行 `python test_llama_server.py` 查看详细错误
3. 访问 llama.cpp GitHub 仓库查看 issue
4. 提供完整的错误信息和日志
