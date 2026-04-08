# ComfyUI Llama-YF v1.3 更新版本
# 基于 v1.0 增强版
# 版本: v1.3.0
# 创建日期: 2026-04-08

# 核心代码文件（v1.3）
nodes.py
__init__.py

# 依赖库文件（llama.cpp 编译产物）
llama/llama-mtmd-cli.exe
llama/llama-server.exe
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

# 不包含的文件（开发/测试文件）
# test/ (测试文件)
# log/ (日志)
# *.backup.* (备份文件)
# *.md.backup.* (文档备份)
# UPLOAD_FILES.txt
# UPLOAD_FILES.md
# UPLOAD_FILES_v1.0.txt
# UPLOAD_FILES_v1.3.txt

# v1.3 新增功能
# - API 模式支持（llama.cpp server）
# - 模块化节点（LlamaModelSelect, LlamaParams, LlamaVideoParams）
# - 视频参数独立节点
# - 多图支持（images/video 模式）
# - 思考链提取（enable_thinking）
# - 全面中文化（12 种预设提示词）
# - 本地编译架构优化
# - 零 Python 依赖

# v1.3 与 v1.0 的区别
# - 新增 llama-server.exe（API 模式）
# - 新增 LlamaVideoParams 节点
# - 新增模块化节点支持
# - 新增多图批量处理
# - 新增思考链提取
# - 新增全面中文化