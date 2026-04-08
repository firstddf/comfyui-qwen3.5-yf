# ComfyUI Llama-YF v1.0 原始版本（来自 GitHub）
# 项目地址: https://github.com/firstddf/comfyui-qwen3.5-yf
# 版本: v1.0.0
# 创建日期: 2026-04-08

# 核心代码文件（v1.0）
nodes.py
__init__.py

# 依赖库文件（llama.cpp 编译产物）
llama/llama-mtmd-cli.exe
llama/ggml-base.dll
llama/ggml-cpu.dll
llama/ggml-cuda.dll
llama/ggml.dll
llama/llama.dll
llama/mtmd.dll

# 配置文件
.gitignore
LICENSE

# 文档文件
README.md

# 不包含的文件（v1.0 不包含）
# llama/llama-server.exe (API 模式)
# test/ (测试文件)
# log/ (日志)

# v1.0 功能特性
# - 基础 Qwen3.5 GGUF 支持
# - 多模态图像处理
# - GPU 加速支持
# - 本地编译架构（不依赖 llama-cpp-python）

# v1.0 不包含的功能
# - API 模式支持（llama.cpp server）
# - 模块化节点（LlamaModelSelect, LlamaParams, LlamaVideoParams）
# - 视频参数独立节点
# - 多图支持（images/video 模式）
# - 思考链提取（enable_thinking）
# - 全面中文化