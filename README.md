# ComfyUI Llama-YF 本地编译 Llama 插件

**版本**: v1.3.0

一个用于在 ComfyUI 中运行多模态 LLM 模型的插件，基于本地编译的 llama.cpp，**不依赖 llama-cpp-python**。

## 🚀 核心优势

- **📦 零 Python 依赖** - 不依赖 `llama-cpp-python`，避免依赖冲突
- **⚡ 更快更新** - 直接更新二进制文件，无需等待 Python 包发布
- **🔧 更易维护** - 编译环境独立，兼容性更强
- **🔄 最新功能** - 可直接集成 llama.cpp 最新特性
- **🌐 API 模式支持** - 支持 llama.cpp server 模式，可远程调用

## 功能特性

- ✅ 支持多模态模型 (图像 + 文本)
- ✅ 自动扫描 ComfyUI 的 `models/LLM` 目录（支持子目录）
- ✅ 支持 GGUF 格式模型文件和 mmproj 视觉投影
- ✅ 可调节的推理参数
- ✅ GPU/CPU 加速支持（可切换）
- ✅ 简洁的输出日志
- ✅ 12 种中文预设提示词模板
- ✅ 视频/多图处理支持
- ✅ 三种推理模式（one by one / images / video）
- ✅ 思考链提取（think 标签解析）
- ✅ API 模式（llama.cpp server）
- ✅ 模块化节点支持（模型选择 + 参数配置 + 推理）

## 🔧 本地编译架构优势

| 特性       | 传统方法 (llama-cpp-python) | 本地编译架构    |
| -------- | ----------------------- | --------- |
| **依赖管理** | 依赖 Python 包             | 独立可执行文件   |
| **更新速度** | 需等待 Python 包更新          | 直接更新二进制文件 |
| **兼容性**  | 易受 Python 版本影响          | 编译环境独立    |
| **性能**   | Python 层开销              | 直接原生执行    |
| **灵活性**  | 受限于 Python 包功能          | 可直接集成最新特性 |

## 编译信息

**llama.cpp 编译环境**:

- **显卡**: Tesla V100 (sm\_70 架构)
- **CUDA**: 12.8
- **Python**: 3.10
- **编译器**: Visual Studio 2022 BuildTools (MSVC 14.44.35207)
- **架构**: x64, sm\_70 (Tesla V100 计算能力 7.0)

## 🎯 零依赖安装

### 安装要求

1. **无需安装 Python 包**
   - 本插件已包含预编译的 `llama-mtmd-cli.exe` 可执行文件
   - 不需要安装 `llama-cpp-python` 或任何其他 Python 依赖
2. **下载模型文件**：
   - 将 GGUF 模型文件放入 `ComfyUI/models/LLM` 目录
   - 需要同时包含 `.gguf` 模型文件和对应的 `.gguf` 视觉投影文件 (mmproj)
3. **模型文件结构示例**：
   ```
   ComfyUI/models/LLM/
   └── qwen3.5llm/
       ├── Qwen3.5-4B-Q4_K_M.gguf
       └── 3.5mmproj-BF16.gguf
   ```

## 🔄 如何更新到最新版本

### 快速更新步骤

1. **下载最新 llama.cpp**
   - 访问 [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
   - 或自行编译最新版本
2. **替换二进制文件**
   - 将新的 `llama-mtmd-cli.exe` 复制到 `llama/` 目录
   - 无需更新插件其他文件
3. **重启 ComfyUI**
   - 新版本立即生效

### 更新优势

- ✅ **无需等待 Python 包发布**
- ✅ **可直接集成最新 llama.cpp 特性**
- ✅ **更新过程简单快速**
- ✅ **降低兼容性风险**

## 运行环境要求

- **操作系统**: Windows 10/11 x64
- **CUDA**: 12.8 (必须与编译版本匹配)
- **显卡驱动**: ≥ 552.22 (支持 CUDA 12.8)
- **显卡**: Tesla V100 或兼容 sm\_70 架构的显卡
- **显存**: 至少 8GB (推荐 16GB+)

## 使用方法

### 方式一：单一集成节点（推荐）

1. **安装插件**：
   将此插件文件夹复制到 `ComfyUI/custom_nodes/` 目录
2. **重启 ComfyUI**
3. **在 ComfyUI 中使用节点**：
   - 搜索 "llama-yf" 节点
   - 选择模型文件和视觉投影文件
   - 选择预设提示词或输入自定义提示
   - 上传图像（可选）
   - 调节参数并运行

### 方式二：模块化节点组合

**四个独立节点可组合使用**：

1. **LlamaModelSelect** - 模型选择节点
   - 选择模型和 mmproj 文件
   - 配置 API 设置
   - 输出：`model_info`
2. **LlamaParams** - 参数配置节点
   - 配置推理参数
   - 输出：`params_info`
3. **LlamaVideoParams** - 视频参数节点（新增）
   - 配置视频处理参数
   - **仅在 video 模式下生效**
   - 输出：`video_params_info`
4. **LlamaInference** - 推理执行节点
   - 接收 `model_info`、`params_info` 和 `video_params_info`
   - 执行推理
   - 输出：`RESPONSE`, `THINKING`, `OUTPUT_LIST`

**优势**：

- 可复用模型和参数配置
- 工作流更清晰
- 适合复杂场景
- 视频参数独立管理

## 参数说明

### 基本参数

- **model\_file**: 选择主模型文件 (.gguf)
- **mmproj\_file**: 选择视觉投影文件 (mmproj)
- **preset\_prompt**: 预设提示词模板（12 种中文选项）
- **custom\_prompt**: 自定义提示词（覆盖预设）
- **system\_prompt**: 系统提示词（支持中文）

### 推理模式

- **inference\_mode**:
  - `one by one` - 逐图处理（稳定，每张图单独输出）
  - `images` - 多图批量处理（所有图像一起推理）
  - `video` - 视频帧采样处理（均匀采样 max\_frames 帧）

### 视频/多图参数

- **max\_frames**: 视频模式最大采样帧数（默认：24，范围：2-1024）
- **max\_size**: 图像最大尺寸（默认：256，范围：128-16384）
  - 大于此尺寸的图像会自动缩放
  - 保持宽高比

### 推理参数

- **max\_tokens**: 最大生成 token 数量（默认：4096，范围：64-32768）
- **temperature**: 温度参数（默认：0.6，范围：0.0-2.0）
  - 0.6-0.7 推荐用于图像描述
- **top\_p**: 核采样阈值（默认：0.9，范围：0.0-1.0）
- **top\_k**: Top-K 采样（默认：40，范围：1-100）
- **repeat\_penalty**: 重复惩罚（默认：1.0，范围：0.5-2.0）
- **n\_gpu\_layers**: GPU 加载层数（默认：99，范围：-1-200）
  - -1 或 99 表示卸载所有层到 GPU
- **ctx\_size**: 上下文大小（默认：4096，范围：1024-131072）
  - GPU 模式下建议不超过 4096 以防止 stack overflow
- **enable\_thinking**: 启用思考链输出（默认：False）
  - True: 输出包含 `<think>` 标签的思考过程
- **seed**: 随机种子（默认：-1，表示随机）
- **threads**: CPU 线程数（默认：-1，自动）

### API 模式参数

- **use\_api**: 使用本地 API 模式（默认：False）
  - True: 使用 llama.cpp server (127.0.0.1:8080)
  - False: 使用 llama-mtmd-cli 本地执行
- **api\_url**: API 服务器地址（默认：<http://127.0.0.1:8080）>
- **api\_model**: 模型名称（需与服务器配置匹配，默认：llama）

### 高级调试参数

- **disable\_warmup**: 禁用 warmup（默认：True）
  - 避免某些 CUDA 初始化崩溃
- **fit\_off**: 禁用参数拟合（默认：True）
  - 减轻某些 GPU 分配 bug
- **force\_cpu**: 强制 CPU 推理（默认：True）
  - 避免 CUDA 问题，但速度较慢

## 📁 文件结构

```
comfyui-llama-yf/
├── __init__.py              # 插件注册
├── nodes.py                 # 主节点代码
├── llama/
│   ├── llama-mtmd-cli.exe   # 本地编译的推理工具
│   └── llama-server.exe     # llama.cpp 服务器（用于 API 模式）
├── README.md                # 本文档
└── models/                  # 模型存储目录（需创建）
```

## 预设提示词

插件内置 12 种中文预设提示词模板：

| 预设名称        | 用途           |
| ----------- | ------------ |
| 空 - 无       | 不使用预设        |
| 常规 - 描述     | 描述图像/视频内容    |
| 提示风格 - 标签   | 生成标签列表用于文生图  |
| 提示风格 - 简洁   | 生成简洁的文生图提示   |
| 提示风格 - 详细   | 生成详细的文生图提示   |
| 提示风格 - 极度详细 | 生成极其详细的文生图提示 |
| 提示风格 - 电影感  | 生成电影感的文生图提示  |
| 创意 - 详细分析   | 详细分析主体、背景、构图 |
| 创意 - 视频总结   | 总结视频关键事件和叙事  |
| 创意 - 短篇故事   | 基于图像/视频写短篇故事 |
| 创意 - 精炼与扩展  | 精炼和增强用户提示    |
| 视觉 - \*边界框  | 定位目标类别的边界框   |

## 输出说明

节点输出三个值：

1. **RESPONSE**: 主要回答内容
2. **THINKING**: 思考过程（如果启用 enable\_thinking）
3. **OUTPUT\_LIST**: 输出列表
   - one by one 模式：包含每张图的独立输出
   - images/video 模式：包含单个输出

## 注意事项

- 模型和视觉投影文件必须匹配（嵌入维度相同）
- 如果遇到内存不足错误，可尝试降低 `n_gpu_layers` 或 `ctx_size`
- 推荐使用 16GB+ 显存的 GPU 运行
- 图像处理可能需要较长时间，请耐心等待
- 必须使用与编译环境匹配的 CUDA 版本 (12.8)
- GPU 模式下 `ctx_size` 超过 4096 会自动限制以防止 stack overflow
- 视频模式会均匀采样 `max_frames` 帧进行处理
- one by one 模式最稳定，适合大量图像
- images/video 模式速度快，但可能遇到内存限制
- API 模式需要手动启动 llama-server 或设置 use\_api=True 自动启动

## 常见问题

**Q: 插件和 llama-cpp-python 有什么区别？**

A: 本插件使用本地编译的 llama.cpp 可执行文件，不依赖 Python 包。优势包括：

- 零 Python 依赖，避免依赖冲突
- 更新更快，直接替换二进制文件
- 编译环境独立，兼容性更强
- 可直接集成 llama.cpp 最新特性
- 支持 API 模式，可远程调用

**Q: 模型找不到怎么办？**

A: 确保模型文件在 `ComfyUI/models/LLM/` 目录下，插件会自动扫描子目录。

**Q: 出现内存不足错误？**

A: 降低 `n_gpu_layers` 或 `ctx_size` 参数值。GPU 模式下建议 `ctx_size` 不超过 4096。

**Q: 输出总是相同？**

A: 检查 `seed` 参数，设为 -1 可获得随机输出。

**Q: CUDA 加速无效？**

A: 确保安装了匹配的 CUDA 12.8 和相应显卡驱动。或设置 `force_cpu=True` 使用 CPU 推理。

**Q: 视频模式如何处理输入？**

A: 视频模式通过 `max_frames` 参数从输入图像中均匀采样帧数。所有采样帧会被一起传递给模型进行推理。

**Q: one by one 和 images 模式有什么区别？**

A:

- **one by one**: 每张图单独推理，输出独立结果，稳定性高
- **images**: 所有图一起推理，模型将多图视为一个序列，适合多图关联分析

**Q: 如何启用思考模式？**

A: 设置 `enable_thinking=True`，模型会在 `<think>` 标签内输出思考过程，最终答案在 `</think>` 标签后。

**Q: API 模式如何使用？**

A:

1. 设置 `use_api=True`
2. 确保 llama-server 在 127.0.0.1:8080 运行，或设置 `api_url` 指定地址
3. 插件会自动启动服务器（如果未运行）

**Q: 如何更新到最新 llama.cpp？**

A: 下载最新编译的 `llama-mtmd-cli.exe` 并替换 `llama/` 目录中的文件即可，无需更新插件其他部分。

## 致谢

本插件在开发过程中参考并借用了以下项目的代码和思路，特此感谢：

- **[lihaoyun6/ComfyUI-llama-cpp\_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm)** - 参考了其预设提示词系统（`PRESET_PROMPTS`）和视频处理逻辑（`inference_mode`、`max_frames`、`max_size` 等参数）
- **Time-AI 视频博主** - 感谢其优秀的 ComfyUI 教程视频，特别是关于 AI 漫剧/短剧自动写剧情的工作流分享

  <https://www.bilibili.com/video/BV1WfwezZET7/?spm_id_from=333.1391.0.0&vd_source=86e36bf6a2dad93d8bd4069941237c10>
- **llama.cpp 社区** - 提供了强大的多模态推理基础
- **ComfyUI 生态** - 优秀的节点化工作流框架

特别感谢原作者的优秀工作，使我们能够在此基础上构建更优化的本地编译架构。

## 📝 更新日志

### v1.3.0 (当前版本)

- 🌐 **API 模式支持** - 新增 llama.cpp server 模式，支持远程调用
- 🔧 **模块化节点** - 新增 LlamaModelSelect、LlamaParams、LlamaInference 三个独立节点
- **视频参数节点** - 新增 LlamaVideoParams 节点，独立管理视频参数（max\_frames、max\_size）
- 🎯 **参数优化** - 调整默认参数值，优化推理稳定性
- 🐛 **Bug 修复** - 修复 GPU 内存分配问题，添加 force\_cpu 选项
- 📝 **文档完善** - 更新 API 模式使用说明和模块化节点组合方法

### v1.2.0 (近期更新)

- 🏗️ **本地编译架构** - 重构为不依赖 llama-cpp-python 的独立编译架构
- 🎯 **节点重命名** - Qwen35GGUF 节点重命名为 `llama-yf`
- 🇨🇳 **全面中文化** - 12 种预设提示词模板翻译为中文
- 🎥 **视频处理修复** - 修复 video 模式多图处理，支持所有采样帧
- 🔧 **架构优化** - 强调本地编译优势，更快更容易更新到最新
- 📝 **致谢添加** - 感谢 lihaoyun6 和 Time-AI 视频博主的贡献

### v1.0.0 (初始版本)

- ✅ 基础 Qwen3.5 GGUF 支持
- ✅ 多模态图像处理
- ✅ GPU 加速支持

## 许可证

MIT License
