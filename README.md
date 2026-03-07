# ComfyUI Qwen3.5 GGUF Plugin

**版本**: v1.0.0

一个用于在 ComfyUI 中运行 Qwen3.5 模型的插件，基于 llama.cpp 的多模态功能。

## 功能特性

- ✅ 支持 Qwen3.5 多模态模型 (图像 + 文本)
- ✅ 自动扫描 ComfyUI 的 `models/LLM` 目录
- ✅ 支持 GGUF 格式模型文件
- ✅ 支持子目录模型文件
- ✅ 可调节的推理参数
- ✅ GPU 加速支持
- ✅ 简洁的输出日志

## 编译信息

**llama.cpp 编译环境**:
- **显卡**: Tesla V100 (sm_70 架构)
- **CUDA**: 12.8
- **Python**: 3.10
- **编译器**: Visual Studio 2022 BuildTools (MSVC 14.44.35207)
- **架构**: x64, sm_70 (Tesla V100 计算能力 7.0)

## 安装要求

1. **下载模型文件**：
   - 将 Qwen3.5 GGUF 模型文件放入 `ComfyUI/models/LLM` 目录
   - 需要同时包含 `.gguf` 模型文件和对应的 `.gguf` 视觉投影文件 (mmproj)

2. **模型文件结构示例**：
   ```
   ComfyUI/models/LLM/
   └── qwen3.5llm/
       ├── Qwen3.5-4B-Q4_K_M.gguf
       └── 3.5mmproj-BF16.gguf
   ```

## 运行环境要求

- **操作系统**: Windows 10/11 x64
- **CUDA**: 12.8 (必须与编译版本匹配)
- **显卡驱动**: ≥ 552.22 (支持 CUDA 12.8)
- **显卡**: Tesla V100 或兼容 sm_70 架构的显卡
- **显存**: 至少 8GB (推荐 16GB+)

## 使用方法

1. **安装插件**：
   将此插件文件夹复制到 `ComfyUI/custom_nodes/` 目录

2. **重启 ComfyUI**

3. **在 ComfyUI 中使用节点**：
   - 节点名称：`Qwen 3.5 (GGUF)`
   - 选择模型文件和视觉投影文件
   - 输入提示词和上传图像
   - 调节参数并运行

## 参数说明

- **model_file**: 选择主模型文件 (.gguf)
- **mmproj_file**: 选择视觉投影文件 (mmproj)
- **prompt**: 用户提示词
- **system_prompt**: 系统提示词
- **max_tokens**: 最大生成 token 数量 (默认：1024)
- **temperature**: 温度参数 (默认：0.7)
- **top_p**: 核采样阈值 (默认：0.9)
- **top_k**: Top-K 采样 (默认：40)
- **repeat_penalty**: 重复惩罚 (默认：1.1)
- **n_gpu_layers**: GPU 加载层数 (默认：28)
- **ctx_size**: 上下文大小 (默认：2048)
- **seed**: 随机种子 (默认：-1, 表示随机)
- **threads**: CPU 线程数 (默认：-1, 自动)

## 注意事项

- 模型和视觉投影文件必须匹配（嵌入维度相同）
- 如果遇到内存不足错误，可尝试降低 `n_gpu_layers` 或 `ctx_size`
- 推荐使用 16GB+ 显存的 GPU 运行
- 图像处理可能需要较长时间，请耐心等待
- 必须使用与编译环境匹配的 CUDA 版本 (12.8)

## 常见问题

**Q: 模型找不到怎么办？**
A: 确保模型文件在 `ComfyUI/models/LLM/` 目录下，插件会自动扫描子目录。

**Q: 出现内存不足错误？**
A: 降低 `n_gpu_layers` 或 `ctx_size` 参数值。

**Q: 输出总是相同？**
A: 检查 `seed` 参数，设为 -1 可获得随机输出。

**Q: CUDA 加速无效？**
A: 确保安装了匹配的 CUDA 12.8 和相应显卡驱动。

## 许可证

MIT License