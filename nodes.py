# ComfyUI-llama-yf Plugin
# Fast inference node using llama.cpp (via llama-mtmd-cli subprocess)
# Model files are loaded from ComfyUI's models/LLM directory
#
# Credits and Acknowledgments:
# - Preset prompt system (PRESET_PROMPTS) and video processing logic (inference_mode,
#   max_frames, max_size) are adapted from lihaoyun6/ComfyUI-llama-cpp_vlm
#   (https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm)
# - Thanks to Time-AI video creator for excellent ComfyUI tutorials and workflow sharing
# - Thanks to the llama.cpp community for the multimodal inference foundation
# - Built upon the ComfyUI ecosystem for node-based workflow

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import random
import io
import base64

import numpy as np
import torch
from PIL import Image
import requests

import folder_paths
import comfy.model_management as mm

THINK_BLOCK_RE = re.compile(
    r"<think[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL
)

# Preset prompts from nodes-lihaoyun6.py
PRESET_PROMPTS = {
    "空 - 无": "",
    "常规 - 描述": "描述这个@。",
    "提示风格 - 标签": "你的任务是为文生@AI生成一个简洁的、用逗号分隔的标签列表，仅基于@中的视觉信息。输出最多50个唯一标签。严格描述视觉元素，如主体、服装、环境、颜色、光线和构图。不要包含抽象概念、解释、营销术语或技术行话（例如，不要包含'SEO'、'品牌对齐'、'病毒潜力'）。目标是简洁的视觉描述列表。避免重复标签。",
    "提示风格 - 简洁": "分析@并生成一个简单的、单句的文生@提示。简洁地描述主要主体和场景。",
    "提示风格 - 详细": "基于@生成一个详细、艺术的文生@提示。将主体、其动作、环境、光线和整体氛围结合成一个连贯的段落，大约2-3句话。专注于关键视觉细节。",
    "提示风格 - 极度详细": "从@生成一个极其详细和描述性的文生@提示。创建一个丰富的段落，详细阐述主体的外观、服装纹理、具体的背景元素、光线的质量和颜色、阴影以及整体氛围。目标是高度描述性和沉浸式的提示。",
    "提示风格 - 电影感": "作为主提示工程师。为@生成AI创建一个高度详细且引人入胜的提示。描述主体、其姿势、环境、光线、氛围和艺术风格（例如，照片级真实感、电影感、绘画感）。将所有元素编织成一个自然的语言段落，注重视觉冲击力。",
    "创意 - 详细分析": "详细描述这个@，将主体、着装、配饰、背景和构图分解为独立的章节。",
    "创意 - 视频总结": "总结此视频中的关键事件和叙事要点。",
    "创意 - 短篇故事": "基于这个@或视频写一个简短的、富有想象力的故事。",
    "创意 - 精炼与扩展提示": "为用户提示进行精炼和增强，用于创意文生@生成。保持含义和关键词，使其更具表现力和视觉丰富性。仅输出改进后的提示文本本身，不要包含任何推理步骤、思考过程或额外评论。",
    "视觉 - *边界框": '定位属于以下类别的每个实例："#"。以列表形式报告边界框坐标，JSON格式为{{"bbox_2d": [x1, y1, x2, y2], "label": "string"}}。'
}

PRESET_TAGS = list(PRESET_PROMPTS.keys())


def image2base64(image):
    """Convert image tensor to base64 string"""
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64


def scale_image(image: torch.Tensor, max_size: int = 128):
    """Scale image to max_size while preserving aspect ratio"""
    img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    w, h = img_pil.size
    scale = min(max_size / max(w, h), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return np.array(img_resized)


def get_model_files():
    """扫描 LLM 目录下的所有 GGUF 模型文件（包括子目录）"""
    llm_dir = Path(folder_paths.models_dir) / "LLM"
    if not llm_dir.exists():
        llm_dir.mkdir(parents=True, exist_ok=True)
        return ["-- 请先将模型文件放入 models/LLM 目录 --"]
    
    # 递归扫描所有子目录中的 GGUF 文件
    gguf_files = []
    for f in llm_dir.rglob("*.gguf"):
        if not f.name.startswith("mmproj"):
            # 使用相对路径作为标识，方便定位
            rel_path = f.relative_to(llm_dir)
            gguf_files.append(str(rel_path))
    
    if not gguf_files:
        return ["-- 未找到 GGUF 模型文件 --"]
    
    return sorted(gguf_files)


def get_mmproj_files():
    """扫描 LLM 目录下的所有 mmproj 文件（包括子目录）"""
    llm_dir = Path(folder_paths.models_dir) / "LLM"
    if not llm_dir.exists():
        return ["-- 请先将模型文件放入 models/LLM 目录 --"]
    
    # 递归扫描所有子目录中的 mmproj 文件
    mmproj_files = []
    for f in llm_dir.rglob("*.gguf"):
        if "mmproj" in f.name.lower():
            # 使用相对路径作为标识，方便定位
            rel_path = f.relative_to(llm_dir)
            mmproj_files.append(str(rel_path))
    
    if not mmproj_files:
        return ["-- 未找到 mmproj 文件 --"]
    
    return sorted(mmproj_files)


class LlamaYF:
    """LLaMA YF node — fast inference via llama.cpp."""

    @classmethod
    def INPUT_TYPES(cls):
        model_files = get_model_files()
        mmproj_files = get_mmproj_files()
        return {
            "required": {
                "model_file": (get_model_files(), {
                    "default": model_files[0] if model_files and model_files[0] != "-- 请先将模型文件放入 models/LLM 目录 --" and model_files[0] != "-- 未找到 GGUF 模型文件 --" else "",
                    "tooltip": "选择 GGUF 模型文件",
                }),
                "mmproj_file": (get_mmproj_files(), {
                    "default": mmproj_files[0] if mmproj_files and mmproj_files[0] != "-- 请先将模型文件放入 models/LLM 目录 --" and mmproj_files[0] != "-- 未找到 mmproj 文件 --" else "",
                    "tooltip": "选择 mmproj 文件",
                }),
                "preset_prompt": (PRESET_TAGS, {
                    "default": "常规 - 描述",
                    "tooltip": "选择预设提示词模板。使用 custom_prompt 可覆盖。",
                }),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "If provided, this will override the preset prompt",
                    "tooltip": "Custom prompt to override preset. If empty, uses preset_prompt.",
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional system prompt to set model behavior (支持中文)",
                }),
                "inference_mode": (["one by one", "images", "video"], {
                    "default": "one by one",
                    "tooltip": "one by one: Process one image at a time\nimages:  Process all images at once\nvideo:  Treat input images as video frames"
                }),
                "max_frames": ("INT", {
                    "default": 24,
                    "min": 2,
                    "max": 1024,
                    "step": 1,
                    "tooltip": 'Number of frames to sample evenly from input video (for "video" mode only)'
                }),
                "max_size": ("INT", {
                    "default": 256,
                    "min": 128,
                    "max": 16384,
                    "step": 64,
                    "tooltip": 'Max size (in pixels) for input images. Images larger than this will be scaled down while preserving aspect ratio. Applies to all inference modes.'
                }),
                "max_tokens": ("INT", {
                    "default": 4096,
                    "min": 64,
                    "max": 32768,
                    "tooltip": "Maximum tokens to generate",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature (0.6-0.7 recommended for captioning)",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling threshold",
                }),
                "top_k": ("INT", {
                    "default": 40,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Top-K sampling",
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Penalty for repeated tokens",
                }),
                "n_gpu_layers": ("INT", {
                    "default": 99,
                    "min": -1,
                    "max": 200,
                    "tooltip": "-1 or 99 offloads all layers to GPU",
                }),
                "ctx_size": ("INT", {
                    "default": 4096,
                    "min": 1024,
                    "max": 131072,
                    "step": 1024,
                    "tooltip": "Context window size in tokens",
                }),
                "use_api": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use local API (127.0.0.1:8080) instead of llama-mtmd-cli",
                }),
                "api_url": ("STRING", {
                    "default": "http://127.0.0.1:8080",
                    "tooltip": "Local llama.cpp server address",
                }),
                "api_model": ("STRING", {
                    "default": "llama",
                    "tooltip": "Model name for API mode (must match server configuration)",
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable thinking mode. Outputs reasoning in THINKING output.",
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed for reproducibility (-1 = random)",
                }),
                "threads": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 64,
                    "tooltip": "Number of CPU threads (-1 = auto, recommended)",
                }),
                "disable_warmup": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable llama-mtmd-cli warmup (--no-warmup) to avoid some CUDA init crashes.",
                }),
                "fit_off": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable params fit (-fit off) to mitigate some GPU allocation bugs.",
                }),
                "force_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Force CPU inference instead of GPU to avoid CUDA issues.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Image for vision tasks"}),
                "cli_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to llama-mtmd-cli binary. Auto-detected if empty.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "THINKING", "OUTPUT_LIST")
    OUTPUT_IS_LIST = (False, False, True)
    FUNCTION = "process"
    CATEGORY = "llama-yf"

    @classmethod
    def IS_CHANGED(cls, model_file, mmproj_file, *args, **kwargs):
        use_api = kwargs.get('use_api', False)
        if use_api:
            return ""  # API模式下不需要检查本地文件变化
        
        # 当模型文件改变时重新计算
        llm_dir = Path(folder_paths.models_dir) / "LLM"
        model_path = llm_dir / model_file
        mmproj_path = llm_dir / mmproj_file
        
        mtime = 0
        if model_path.exists():
            mtime += model_path.stat().st_mtime
        if mmproj_path.exists():
            mtime += mmproj_path.stat().st_mtime
        
        return mtime

    @staticmethod
    def _ensure_model(model_file: str, mmproj_file: str) -> tuple[Path, Path]:
        """Load model files from ComfyUI's models/LLM directory"""
        
        # 使用 ComfyUI 的标准模型目录
        base_models_dir = Path(folder_paths.models_dir)
        llm_dir = base_models_dir / "LLM"
        
        # 创建 LLM 目录如果不存在
        llm_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用相对路径构建完整路径（支持子目录）
        model_path_obj = llm_dir / model_file
        mmproj_path_obj = llm_dir / mmproj_file
        
        # 验证模型文件是否存在
        if not model_path_obj.exists():
            files = [str(f.relative_to(llm_dir)) for f in llm_dir.rglob("*.gguf") if not f.name.startswith("mmproj")]
            file_list = "\n  ".join(files)
            raise FileNotFoundError(
                f"[llama-yf] Model file not found: {model_path_obj}\n"
                f"Available GGUF files in {llm_dir}:\n  {file_list}"
            )
        
        if not mmproj_path_obj.exists():
            files = [str(f.relative_to(llm_dir)) for f in llm_dir.rglob("*.gguf") if "mmproj" in f.name.lower()]
            file_list = "\n  ".join(files)
            raise FileNotFoundError(
                f"[llama-yf] MMProj file not found: {mmproj_path_obj}\n"
                f"Available mmproj files in {llm_dir}:\n  {file_list if file_list else '-- 无 --'}"
            )
        
        print(f"[llama-yf] Using model: {model_path_obj}")
        print(f"[llama-yf] Using mmproj: {mmproj_path_obj}")
        
        return model_path_obj, mmproj_path_obj

    @staticmethod
    def _find_cli(cli_path_override: str) -> str:
        """Find the llama-mtmd-cli binary - 优先使用插件目录内的版本"""
        
        # 1. 优先使用用户通过参数指定的路径
        if cli_path_override and cli_path_override.strip():
            p = Path(cli_path_override.strip())
            if p.is_file() and os.access(str(p), os.X_OK):
                print(f"[llama-yf] Using specified CLI: {p}")
                return str(p)
            else:
                print(f"[llama-yf] Warning: Specified CLI not found: {p}")
        
        # 2. 使用插件目录内的llama-mtmd-cli.exe
        plugin_dir = Path(__file__).parent
        plugin_cli = plugin_dir / "llama" / "llama-mtmd-cli.exe"
        if plugin_cli.is_file() and os.access(str(plugin_cli), os.X_OK):
            print(f"[llama-yf] Using plugin CLI: {plugin_cli}")
            return str(plugin_cli)
        
        # 3. 检查当前目录的llama子目录
        local_llama = Path.cwd() / "llama" / "llama-mtmd-cli.exe"
        if local_llama.is_file() and os.access(str(local_llama), os.X_OK):
            print(f"[llama-yf] Using local llama CLI: {local_llama}")
            return str(local_llama)
        
        # 4. 最后才去PATH找
        found = shutil.which("llama-mtmd-cli")
        if found:
            print(f"[llama-yf] ⚠️  Using PATH CLI: {found}")
            return found
        
        raise FileNotFoundError(
            "[llama-yf] llama-mtmd-cli not found. "
            "Please compile it or set the cli_path input. "
            "The binary should be in the plugin's llama/ directory."
        )

    @staticmethod
    def _tensor_to_temp_image(tensor: torch.Tensor) -> str:
        """Save ComfyUI IMAGE tensor as a temporary PNG. Returns file path."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(array)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        pil.save(path, format="PNG")
        return path

    @staticmethod
    def _invoke_cli(
        cli_path: str,
        model_path: Path,
        mmproj_path: Path,
        prompt: str,
        system_prompt: str,
        image_paths: str | list[str] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        n_gpu_layers: int,
        ctx_size: int,
        enable_thinking: bool,
        seed: int,
        threads: int,
        disable_warmup: bool = False,
        fit_off: bool = False,
        max_frames: int = -1,
        force_cpu: bool = False,
    ) -> str:
        """Run llama-mtmd-cli and return the generated text."""
        
        # 如果 seed 为 -1，则生成随机种子
        if seed == -1:
            actual_seed = random.randint(1, 2**31 - 1)
        else:
            actual_seed = seed
            
        # 基础命令 - 使用 llama.cpp 标准参数
        cmd = [
            cli_path,
            "-m", str(model_path),
            "--mmproj", str(mmproj_path),
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "--top-k", str(top_k),
            "--repeat-penalty", str(repeat_penalty),
            "-ngl", "0" if force_cpu else str(n_gpu_layers),
            "-c", str(ctx_size),
            "--seed", str(actual_seed),
            "-t", str(threads),
            "--image-min-tokens", "1024",  # Qwen-VL 需要至少 1024 图像 tokens
            "--verbose",  # 添加详细日志
        ]

        if fit_off:
            cmd.extend(["--fit", "off"])
        if disable_warmup:
            cmd.append("--no-warmup")

        # 如果有图像，添加图像参数
        if image_paths is not None:
            if isinstance(image_paths, str):
                # 单个图像路径
                abs_image_paths = [os.path.abspath(image_paths)]
                print(f"[llama-yf] Using image: {abs_image_paths[0]}")
            else:
                # 多个图像路径列表
                abs_image_paths = [os.path.abspath(p) for p in image_paths]
                print(f"[llama-yf] Using {len(abs_image_paths)} images: {', '.join(os.path.basename(p) for p in abs_image_paths[:3])}" + 
                      ("..." if len(abs_image_paths) > 3 else ""))
            
            # 使用逗号分隔的路径作为 --image 参数值
            cmd.extend(["--image", ",".join(abs_image_paths)])

        # 构建提示词
        if system_prompt and system_prompt.strip():
            full_prompt = f"<|im_start|>system\n{system_prompt.strip()}\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

        # 如果启用了思考模式
        if enable_thinking:
            full_prompt = f"<|im_start|>system\n请先思考再回答，将思考过程放在<think>标签内。\n<|im_end|>\n" + full_prompt

        cmd.extend(["-p", full_prompt])

        # 打印调试信息
        print(f"[llama-yf] Running command: {' '.join(cmd)}")
        print(f"[llama-yf] === 生成参数 ===")
        print(f"[llama-yf]   - 模型: {model_path.name}")
        print(f"[llama-yf]   - 视觉模型: {mmproj_path.name}")
        print(f"[llama-yf]   - 最大 tokens: {max_tokens}")
        print(f"[llama-yf]   - 温度: {temperature}")
        print(f"[llama-yf]   - Top-P: {top_p}")
        print(f"[llama-yf]   - Top-K: {top_k}")
        print(f"[llama-yf]   - 重复惩罚: {repeat_penalty}")
        print(f"[llama-yf]   - GPU层数: {n_gpu_layers}")
        print(f"[llama-yf]   - 上下文: {ctx_size}")
        print(f"[llama-yf]   - 线程数: {threads}")
        print(f"[llama-yf]   - 随机种子: {actual_seed}" if seed != -1 else f"[llama-yf]   - 随机种子: {actual_seed} (random)")
        if image_paths is not None:
            if isinstance(image_paths, str):
                print(f"[llama-yf]   - 图像: {Path(image_paths).name}")
            else:
                image_names = [Path(p).name for p in image_paths]
                if len(image_names) <= 3:
                    print(f"[llama-yf]   - 图像: {', '.join(image_names)}")
                else:
                    print(f"[llama-yf]   - 图像: {', '.join(image_names[:3])}, ... ({len(image_names)} total)")
        print(f"[llama-yf]   - 提示词: {prompt[:50]}..." if len(prompt) > 50 else f"[llama-yf]   - 提示词: {prompt}")
        print(f"[llama-yf] ===================")
        
        try:
            # 运行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=300,  # 5分钟超时
            )

            print(f"[llama-yf] Return code: {result.returncode}")
            
            # 处理 stderr，只显示关键状态信息
            if result.stderr:
                # 只显示关键的 llama.cpp 状态信息
                for line in result.stderr.split('\n'):
                    if 'image slice encoded' in line or 'image decoded' in line or 'decoding image' in line:
                        if 'image slice encoded' in line:
                            print(f"[llama-yf] ✅ 图像编码")
                        elif 'decoding image' in line:
                            print(f"[llama-yf] ✅ 图像解码")
                        elif 'image decoded' in line:
                            print(f"[llama-yf] ✅ 图像处理完成")

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                exit_code_hex = hex(result.returncode & 0xFFFFFFFF)
                print(f"[llama-yf] ❌ 错误输出:")
                print(f"{error_msg}")
                print(f"[llama-yf] Exit code: {result.returncode} ({exit_code_hex})")
                
                # 常见错误提示
                if "CUDA" in error_msg or "cuda" in error_msg or "GPU" in error_msg:
                    print(f"[llama-yf] 💡 提示：检测到 CUDA/GPU 相关错误，请尝试:")
                    print(f"[llama-yf]   - 减少 ctx_size (当前：{ctx_size})")
                    print(f"[llama-yf]   - 减少 max_frames (当前：{max_frames})")
                    print(f"[llama-yf]   - 关闭其他占用显存的程序")
                    print(f"[llama-yf]   - 重启 ComfyUI 释放显存")
                elif "memory" in error_msg.lower() or "alloc" in error_msg.lower():
                    print(f"[llama-yf] 💡 提示：显存不足，请尝试减少参数或重启 ComfyUI")
                
                raise RuntimeError(
                    f"[llama-yf] Inference failed (exit {result.returncode} / {exit_code_hex}): {error_msg}"
                )

            return result.stdout

        except subprocess.TimeoutExpired:
            raise RuntimeError("[llama-yf] Inference timed out after 300 seconds")
        except Exception as e:
            raise RuntimeError(f"[llama-yf] Failed to run inference: {str(e)}")

    @staticmethod
    def _invoke_api(
        api_url: str,
        model: str | None,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        ctx_size: int,
        enable_thinking: bool,
        seed: int,
        image_base64: str | None = None,
        image_base64_list: list[str] | None = None,
    ) -> str:
        """Call local llama.cpp REST API (e.g. 127.0.0.1:8080)."""
        try:
            request_body = {
                "model": model or "llama-yf",
                "prompt": f"{system_prompt + '\n' if system_prompt else ''}{prompt}",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "ctx_size": ctx_size,
                "seed": seed,
                # some llama.cpp server variants may honor this
            }
            if image_base64_list:
                request_body["images"] = image_base64_list
            elif image_base64:
                request_body["image"] = image_base64

            headers = {"Content-Type": "application/json"}
            resp = requests.post(f"{api_url.rstrip('/')}/v1/chat/completions", json=request_body, headers=headers, timeout=300)

            if resp.status_code != 200:
                raise RuntimeError(f"[llama-yf] API call failed ({resp.status_code}): {resp.text}")

            j = resp.json()

            # 支持 openai-like chat 或 completions
            if "choices" in j and len(j["choices"]) > 0:
                first = j["choices"][0]
                if "message" in first and "content" in first["message"]:
                    return first["message"]["content"]
                elif "text" in first:
                    return first["text"]

            if "result" in j:
                return str(j["result"])

            return str(j)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[llama-yf] API request failed: {str(e)}")

    @staticmethod
    def _extract_thinking(text) -> tuple[str, str]:
        """Extract thinking content and clean response. Returns (response, thinking)."""
        if text is None:
            text = ""
        elif not isinstance(text, str):
            try:
                text = str(text)
            except:
                text = ""
        
        thinking = ""

        # Case 1: Complete <think>...</think> block
        match = THINK_BLOCK_RE.search(text)
        if match:
            thinking = re.sub(r"</?think[^>]*>", "", match.group(0)).strip()
            text = THINK_BLOCK_RE.sub("", text).strip()

        # Case 2: </think> without opening tag (stripped by tokenizer)
        elif "</think>" in text:
            parts = text.split("</think>", 1)
            thinking = parts[0].strip()
            text = parts[1].strip()

        # Case 3: <think> without </think> (truncated by max_tokens)
        elif "<think>" in text:
            parts = text.split("<think>", 1)
            before = parts[0].strip()
            thinking = parts[1].strip()
            text = before

        # Clean leftover chat template tokens
        for token in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            text = text.replace(token, "")
        
        return str(text).strip(), str(thinking).strip()

    def process(
        self,
        model_file: str,
        mmproj_file: str,
        preset_prompt: str,
        custom_prompt: str,
        system_prompt: str,
        inference_mode: str,
        max_frames: int,
        max_size: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        n_gpu_layers: int,
        ctx_size: int,
        enable_thinking: bool,
        seed: int,
        threads: int = -1,
        disable_warmup: bool = False,
        fit_off: bool = False,
        force_cpu: bool = False,
        use_api: bool = False,
        api_url: str = "http://127.0.0.1:8080",
        api_model: str = "llama",
        image=None,
        cli_path: str = "",
    ):
        """处理输入并返回模型输出 - 支持 inference_mode"""
        
        print(f"[llama-yf] Starting inference with {model_file}")
        print(f"[llama-yf] Inference mode: {inference_mode}")

        video_input = inference_mode == "video"
        
        # 处理 preset_prompt 逻辑
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
            print(f"[llama-yf] Using custom prompt override")
        else:
            prompt_template = PRESET_PROMPTS[preset_prompt]
            media_type = "视频" if video_input else "图像"
            prompt = prompt_template.replace("@", media_type)
            print(f"[llama-yf] Using preset prompt: {preset_prompt} (media: {media_type})")
        
        # API 模式下不需要本地 llama-mtmd-cli 和模型路径
        cli = None
        model_path = None
        mmproj_path = None
        if not use_api:
            cli = LlamaYF._find_cli(cli_path)
            model_path, mmproj_path = LlamaYF._ensure_model(model_file, mmproj_file)

        out1 = ""  # 单个字符串输出
        out2 = []  # 列表输出
        video_input = inference_mode == "video"
        
        # 如果是 video 模式，添加 system prompt 提示
        if video_input:
            video_hint = "请将输入的图片序列当做视频而不是静态帧序列。"
            if system_prompt:
                system_prompt = video_hint + " " + system_prompt
            else:
                system_prompt = video_hint
        
        try:
            # 处理图像输入
            # 检查是否有有效的图像输入
            has_images = False
            image_list = None
            
            if image is not None:
                # 检查 image 是否为空 tensor 或空列表
                if hasattr(image, '__len__'):
                    if len(image) > 0:
                        has_images = True
                        image_list = image
                elif isinstance(image, torch.Tensor):
                    # 单个图像 tensor，检查是否为空
                    if image.numel() > 0:
                        has_images = True
                        # 将单个 tensor 包装成列表以便统一处理
                        image_list = [image]
            
            # 检查 inference_mode 是否需要图像
            needs_images = inference_mode in ["images", "video"]
            
            if needs_images and not has_images:
                raise ValueError(f"[llama-yf] Inference mode '{inference_mode}' requires image input, but no valid images provided.")
            
            if has_images:
                frames = image_list
                if video_input:
                    # 视频模式：采样帧
                    original_len = len(frames)
                    indices = np.linspace(0, original_len - 1, max_frames, dtype=int)
                    frames = [frames[i] for i in indices]
                    print(f"[llama-yf] 视频模式: 采样 {len(frames)} 帧 from {original_len} 张图像")
                
                if inference_mode == "one by one":
                    # 逐图模式：每张图单独推理
                    tmp_list = []
                    print(f"[llama-yf] Processing {len(frames)} images one by one")
                    
                    for i, img_frame in enumerate(frames):
                        if mm.processing_interrupted():
                            raise mm.InterruptProcessingException()
                        
                        # 如果 max_size 小于原图尺寸，先缩放图像
                        h, w = img_frame.shape[0], img_frame.shape[1]
                        if max(h, w) > max_size:
                            scaled_img = scale_image(img_frame, max_size)
                            scaled_tensor = torch.from_numpy(scaled_img).unsqueeze(0).float() / 255.0
                            image_path = LlamaYF._tensor_to_temp_image(scaled_tensor)
                            print(f"[llama-yf]   Image {i+1}: scaled to {scaled_img.shape[1]}x{scaled_img.shape[0]} (max_size={max_size})")
                        else:
                            image_path = LlamaYF._tensor_to_temp_image(img_frame)
                        
                        try:
                            # 调用 CLI 或 API 进行推理
                            if use_api:
                                with open(image_path, "rb") as f:
                                    image_b64 = base64.b64encode(f.read()).decode("utf-8")
                                raw_output = LlamaYF._invoke_api(
                                    api_url=api_url,
                                    model=api_model,
                                    prompt=prompt,
                                    system_prompt=system_prompt,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=top_k,
                                    repeat_penalty=repeat_penalty,
                                    ctx_size=ctx_size,
                                    enable_thinking=enable_thinking,
                                    seed=seed,
                                    image_base64=image_b64,
                                )
                            else:
                                raw_output = LlamaYF._invoke_cli(
                                    cli_path=cli,
                                    model_path=model_path,
                                    mmproj_path=mmproj_path,
                                    prompt=prompt,
                                    system_prompt=system_prompt,
                                    image_paths=image_path,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=top_k,
                                    repeat_penalty=repeat_penalty,
                                    n_gpu_layers=n_gpu_layers,
                                    ctx_size=ctx_size,
                                    enable_thinking=enable_thinking,
                                    seed=seed,
                                    threads=threads,
                                    disable_warmup=disable_warmup,
                                    fit_off=fit_off,
                                    max_frames=max_frames,
                                    force_cpu=force_cpu,
                            )
                            
                            # 提取思考内容和响应
                            response, thinking = LlamaYF._extract_thinking(raw_output)
                            
                            if not enable_thinking:
                                thinking = ""
                            
                            out2.append(response)
                            if len(frames) > 1:
                                tmp_list.append(f"====== Image {i+1} ======")
                            tmp_list.append(response)
                            
                        finally:
                            # 清理临时文件
                            if image_path and os.path.exists(image_path):
                                os.unlink(image_path)
                    
                    out1 = "\n\n".join(tmp_list)
                    
                else:
                    # images/video 模式：所有图像一起推理
                    print(f"[llama-yf] Processing {len(frames)} images together")
                    
                    # 构建多图提示词 (llama-mtmd-cli 支持多图输入)
                    image_paths = []
                    try:
                        for img_frame in frames:
                            if len(frames) > 1:
                                # 缩放图像
                                scaled_img = scale_image(img_frame, max_size)
                                # 转换为 tensor 格式
                                scaled_tensor = torch.from_numpy(scaled_img).unsqueeze(0).float() / 255.0
                                image_path = LlamaYF._tensor_to_temp_image(scaled_tensor)
                            else:
                                image_path = LlamaYF._tensor_to_temp_image(img_frame)
                            image_paths.append(image_path)
                        
                        # 多图输入：可选择 CLI 或 API 进行推理
                        if use_api:
                            image_b64_list = []
                            for path in image_paths:
                                with open(path, "rb") as f:
                                    image_b64_list.append(base64.b64encode(f.read()).decode("utf-8"))
                            raw_output = LlamaYF._invoke_api(
                                api_url=api_url,
                                model=api_model,
                                prompt=prompt,
                                system_prompt=system_prompt,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                repeat_penalty=repeat_penalty,
                                ctx_size=ctx_size,
                                enable_thinking=enable_thinking,
                                seed=seed,
                                image_base64_list=image_b64_list,
                            )
                        else:
                            raw_output = LlamaYF._invoke_cli(
                                cli_path=cli,
                                model_path=model_path,
                                mmproj_path=mmproj_path,
                                prompt=prompt,
                                system_prompt=system_prompt,
                                image_paths=image_paths if image_paths else None,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                repeat_penalty=repeat_penalty,
                                n_gpu_layers=n_gpu_layers,
                                ctx_size=ctx_size,
                                enable_thinking=enable_thinking,
                                seed=seed,
                                threads=threads,
                                disable_warmup=disable_warmup,
                                fit_off=fit_off,
                                max_frames=max_frames,
                                force_cpu=force_cpu,
                            )
                        
                        response, thinking = LlamaYF._extract_thinking(raw_output)
                        
                        if not enable_thinking:
                            thinking = ""
                        
                        out1 = response
                        out2 = [response]
                        
                    finally:
                        # 清理所有临时文件
                        for image_path in image_paths:
                            if image_path and os.path.exists(image_path):
                                os.unlink(image_path)
            else:
                # 无图像输入，纯文本推理
                print(f"[llama-yf] Text-only inference")
                
                if use_api:
                    raw_output = LlamaYF._invoke_api(
                        api_url=api_url,
                        model=api_model,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repeat_penalty=repeat_penalty,
                        ctx_size=ctx_size,
                        enable_thinking=enable_thinking,
                        seed=seed,
                    )
                else:
                    raw_output = LlamaYF._invoke_cli(
                        cli_path=cli,
                        model_path=model_path,
                        mmproj_path=mmproj_path,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        image_paths=None,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repeat_penalty=repeat_penalty,
                        n_gpu_layers=n_gpu_layers,
                        ctx_size=ctx_size,
                        enable_thinking=enable_thinking,
                        seed=seed,
                        threads=threads,
                        disable_warmup=disable_warmup,
                        fit_off=fit_off,
                        max_frames=max_frames,
                        force_cpu=force_cpu,
                    )
                
                response, thinking = LlamaYF._extract_thinking(raw_output)
                
                if not enable_thinking:
                    thinking = ""
                
                out1 = response
                out2 = [response]

            print(f"[llama-yf] Success! Response length: {len(out1)} chars")
            
            return (out1, thinking, out2)

        except Exception as e:
            print(f"[llama-yf] Error during inference: {str(e)}")
            raise




PRESET_TAGS = list(PRESET_PROMPTS.keys())

def get_model_files():
    llm_dir = Path(folder_paths.models_dir) / "LLM"
    if not llm_dir.exists(): return ["-- 请先将模型文件放入 models/LLM 目录 --"]
    files = [str(f.relative_to(llm_dir)) for f in llm_dir.rglob("*.gguf") if not f.name.startswith("mmproj")]
    return sorted(files) if files else ["-- 未找到 GGUF 模型文件 --"]

def get_mmproj_files():
    """扫描 LLM 目录下的所有 mmproj 文件"""
    llm_dir = Path(folder_paths.models_dir) / "LLM"
    if not llm_dir.exists(): return ["-- 请先将模型文件放入 models/LLM 目录 --"]
    files = [str(f.relative_to(llm_dir)) for f in llm_dir.rglob("*mmproj*.gguf")]
    return sorted(files) if files else ["-- 未找到 mmproj 文件 --"]

class LlamaModelSelect:
    @classmethod
    def INPUT_TYPES(cls):
        files = get_model_files()
        mmproj_files = get_mmproj_files()
        return {"required": {
            "model_file": (files, {"default": files[0] if files and "--" not in files[0] else ""}),
            "mmproj_file": (mmproj_files, {"default": mmproj_files[0] if mmproj_files and "--" not in mmproj_files[0] else ""}),
            "use_api": ("BOOLEAN", {"default": True})
        }, "optional": {"api_url": ("STRING", {"default": "http://127.0.0.1:8080"})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_info",)
    FUNCTION = "select"
    CATEGORY = "llama-yf"
    def select(self, model_file, mmproj_file, use_api, api_url="http://127.0.0.1:8080"):
        model_info = f"{model_file}|{mmproj_file}|{use_api}|{api_url}"
        return (model_info,)

class LlamaParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "max_tokens": ("INT", {"default": 4096, "min": 64, "max": 32768}),
            "ctx_size": ("INT", {"default": 4096, "min": 512, "max": 131072}),
            "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
            "top_k": ("INT", {"default": 40, "min": 0, "max": 200}),
            "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            "enable_thinking": ("BOOLEAN", {"default": False}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("params_info",)
    FUNCTION = "configure"
    CATEGORY = "llama-yf"
    def configure(self, max_tokens, ctx_size, temperature, top_p, top_k, repeat_penalty, seed, enable_thinking):
        params_info = f"{max_tokens}|{ctx_size}|{temperature}|{top_p}|{top_k}|{repeat_penalty}|{seed}|{enable_thinking}"
        return (params_info,)


class LlamaVideoParams:
    """视频处理参数节点 - 仅在 video 模式下生效"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "max_frames": ("INT", {
                "default": 24,
                "min": 2,
                "max": 1024,
                "step": 1,
                "tooltip": "视频模式最大采样帧数（仅在 video 模式下生效）"
            }),
            "max_size": ("INT", {
                "default": 256,
                "min": 128,
                "max": 16384,
                "step": 64,
                "tooltip": "图像最大尺寸，大于此尺寸的图像会自动缩放（保持宽高比）"
            }),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_params_info",)
    FUNCTION = "configure_video"
    CATEGORY = "llama-yf"
    def configure_video(self, max_frames, max_size):
        video_params_info = f"{max_frames}|{max_size}"
        return (video_params_info,)

class LlamaInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preset_prompt": (PRESET_TAGS, {"default": "常规 - 描述"}),
            "custom_prompt": ("STRING", {"default": "", "multiline": True}),
            "system_prompt": ("STRING", {"default": "", "multiline": True}),
            "inference_mode": (["one by one", "images", "video"], {"default": "one by one"}),
            "model_info": ("STRING", {"default": ""}),
            "params_info": ("STRING", {"default": ""}),
            "video_params_info": ("STRING", {"default": ""}),
        }, "optional": {
            "image": ("IMAGE", {}),
        }}
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "THINKING", "OUTPUT_LIST")
    OUTPUT_IS_LIST = (False, False, False)  # 全部输出字符串，不是列表
    FUNCTION = "inference"
    CATEGORY = "llama-yf"
    
    def _find_cli(self):
        for p in [Path(__file__).parent/"llama"/"llama-mtmd-cli.exe", Path("C:/llama.cpp/build/bin/Release/llama-mtmd-cli.exe")]:
            if p.exists(): return str(p)
        raise FileNotFoundError("llama-mtmd-cli.exe not found")
    
    def _ensure_model(self, model_file):
        llm_dir = Path(folder_paths.models_dir) / "LLM"
        model_path = llm_dir / model_file
        if not model_path.exists(): raise FileNotFoundError(f"Model not found: {model_path}")
        mmproj = next((f for f in llm_dir.rglob("*.gguf") if "mmproj" in f.name.lower()), None)
        if not mmproj: raise FileNotFoundError(f"mmproj not found")
        return model_path, mmproj
    
    def _tensor_to_image(self, tensor):
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 4: tensor = tensor.squeeze(0)
            img_np = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            if img_np.shape[0] == 3: img_np = img_np.transpose(1, 2, 0)
            return Image.fromarray(img_np)
        return Image.fromarray(tensor)
    
    def _scale_image(self, tensor, max_size):
        img_np = np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        w, h = img_pil.size
        scale = min(max_size / max(w, h), 1.0)
        return np.array(img_pil.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS))
    
    def _invoke_api(self, api_url, model_file, mmproj_file, ctx_size, prompt, system_prompt, max_tokens, temperature, top_p, top_k, repeat_penalty, seed, enable_thinking=False, image_b64_list=None):
        import subprocess
        import threading
        import time
        import signal
        
        server_process = None
        
        # 检查服务器是否运行
        def is_server_running(url):
            try:
                resp = requests.get(f"{url.rstrip('/')}/v1/models", timeout=2)
                return resp.status_code == 200
            except:
                return False
        
        # 启动服务器
        def start_server(model_file, mmproj_file, ctx_size=4096):
            nonlocal server_process
            llama_dir = Path(__file__).parent / "llama"
            server_exe = llama_dir / "llama-server.exe"
            
            if not server_exe.exists():
                raise FileNotFoundError(f"llama-server.exe not found at {server_exe}")
            
            # 使用用户选择的模型
            llm_dir = Path(folder_paths.models_dir) / "LLM"
            model_path = llm_dir / model_file if model_file else None
            
            # 如果没有指定模型文件，自动查找
            if not model_path or not model_path.exists():
                model_files = list(llm_dir.rglob("*.gguf"))
                model_files = [f for f in model_files if not f.name.startswith("mmproj")]
                if not model_files:
                    raise FileNotFoundError("No GGUF model found")
                model_path = model_files[0]
            
            # 使用用户选择的 mmproj 文件
            if mmproj_file:
                mmproj_path = llm_dir / mmproj_file
            else:
                # 自动查找 mmproj 文件
                mmproj_files = list(model_path.parent.glob("*mmproj*.gguf"))
                if not mmproj_files:
                    mmproj_files = list(llm_dir.rglob("*mmproj*.gguf"))
                if not mmproj_files:
                    raise FileNotFoundError("No mmproj file found")
                mmproj_path = mmproj_files[0]
            
            # 启动服务器
            cmd = [
                str(server_exe),
                "-m", str(model_path),
                "--mmproj", str(mmproj_path),
                "-ngl", "99",
                "-c", str(ctx_size),
                "--port", "8080",
                "--host", "127.0.0.1"
            ]
            print(f"[llama-yf] Server command: {' '.join(cmd)}")
            
            # 启动服务器，输出日志到终端
            server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # 等待服务器启动
            for _ in range(30):
                time.sleep(1)
                if is_server_running(api_url):
                    break
            else:
                raise RuntimeError("Failed to start llama-server")
        
        # 检查是否需要启动服务器
        if not is_server_running(api_url):
            print("[llama-yf] Server not running, starting...")
            try:
                start_server(model_file, mmproj_file, ctx_size)
                print("[llama-yf] Server started")
            except Exception as e:
                print(f"[llama-yf] Failed to start server: {e}")
                raise RuntimeError(f"Failed to start llama-server: {e}")
        
        try:
            resp = requests.get(f"{api_url.rstrip('/')}/v1/models", timeout=5)
            model = resp.json()["data"][0]["id"] if resp.status_code == 200 and "data" in resp.json() else "default"
        except: model = "default"
        
        # 如果启用思考模式，添加特殊的 system prompt
        if enable_thinking:
            if system_prompt and system_prompt.strip():
                system_prompt = "请先思考再回答，将思考过程放在<think>标签内。\n" + system_prompt
            else:
                system_prompt = "请先思考再回答，将思考过程放在<think>标签内。"
        
        messages = []
        if system_prompt and system_prompt.strip(): messages.append({"role": "system", "content": system_prompt.strip()})
        
        # 构建多图输入
        if image_b64_list and len(image_b64_list) > 0:
            user_content = []
            for img_b64 in image_b64_list:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            user_content.append({"type": "text", "text": prompt})
            print(f"[llama-yf] API mode: Processing {len(image_b64_list)} images")
        else:
            user_content = [{"type": "text", "text": prompt}]
            print(f"[llama-yf] API mode: Text-only or single image via legacy method")
        
        messages.append({"role": "user", "content": user_content})
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k, "repeat_penalty": repeat_penalty, "stream": False}
        if seed > 0: body["seed"] = seed
        
        # 打印详细信息
        print(f"[llama-yf] ====== Request Details ======")
        print(f"[llama-yf] Model: {model}")
        print(f"[llama-yf] Max tokens: {max_tokens}")
        print(f"[llama-yf] Temperature: {temperature}")
        print(f"[llama-yf] Top p: {top_p}")
        print(f"[llama-yf] Top k: {top_k}")
        print(f"[llama-yf] Repeat penalty: {repeat_penalty}")
        print(f"[llama-yf] Seed: {seed}")
        print(f"[llama-yf] Enable thinking: {enable_thinking}")
        print(f"[llama-yf] System prompt: {system_prompt[:100] if system_prompt else '(none)'}...")
        print(f"[llama-yf] User prompt: {prompt[:200] if prompt else '(none)'}...")
        print(f"[llama-yf] Has images: {bool(image_b64_list) and len(image_b64_list) > 0}")
        print(f"[llama-yf] =============================")
        
        try:
            resp = requests.post(f"{api_url}/v1/chat/completions", json=body, headers={"Content-Type": "application/json"}, timeout=300)
            print(f"[llama-yf] API response status: {resp.status_code}")
            print(f"[llama-yf] API response: {resp.text[:500]}")
            if resp.status_code != 200: raise RuntimeError(f"API failed ({resp.status_code}): {resp.text[:200]}")
            j = resp.json()
            print(f"[llama-yf] API JSON: {str(j)[:500]}")
            
            # 处理返回内容 - content 是最终答案，reasoning_content 是思考过程
            msg = j.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "")
            reasoning_content = msg.get("reasoning_content", "")
            
            # 处理 enable_thinking 开关
            if enable_thinking:
                # 启用思考模式：content 是答案，reasoning_content 是思考
                result = content if content else reasoning_content
                thinking = reasoning_content
            else:
                # 不启用思考模式：优先使用 content，但如果 content 为空则降级使用 reasoning_content
                if content:
                    result = content
                    thinking = ""
                else:
                    # content 为空时的降级处理（如生成长度超限）
                    result = reasoning_content
                    thinking = ""
            print(f"[llama-yf] Final result length: {len(result)} chars")
            print(f"[llama-yf] Thinking length: {len(thinking)} chars")
        except Exception as e:
            # 如果出错，关闭服务器并重新抛出异常
            if server_process is not None:
                print("[llama-yf] Error occurred, stopping server...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=5)
                except:
                    server_process.kill()
                print("[llama-yf] Server stopped")
            raise e
        
        # 推理成功，关闭服务器
        if server_process is not None:
            print("[llama-yf] Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except:
                server_process.kill()
            print("[llama-yf] Server stopped")
        
        return result, thinking
    
    @staticmethod
    def _extract_thinking(text):
        if text is None: return "", ""
        thinking = ""
        match = THINK_BLOCK_RE.search(text)
        if match:
            thinking = re.sub(r"</?think[^>]*>", "", match.group(0)).strip()
            text = THINK_BLOCK_RE.sub("", text).strip()
        elif "</think>" in text:
            parts = text.split("</think>", 1)
            thinking = parts[0].strip()
            text = parts[1].strip()
        elif "<think>" in text:
            parts = text.split("<think>", 1)
            thinking = parts[1].strip()
            text = parts[0].strip()
        for token in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            text = text.replace(token, "")
        return text.strip(), thinking.strip()
    
    def inference(self, preset_prompt, custom_prompt, system_prompt, inference_mode, model_info="", params_info="", video_params_info="", image=None):
        # 默认值
        model_file = ""
        mmproj_file = ""
        use_api = "True"
        api_url = "http://127.0.0.1:8080"
        max_tokens = 4096
        ctx_size = 4096
        temperature = 0.6
        top_p = 0.9
        top_k = 40
        repeat_penalty = 1.0
        seed = 0
        enable_thinking = False
        max_frames = 24
        max_size = 256
        
        # 解析 model_info (格式: model_file|use_api|api_url)
        if model_info and model_info.strip():
            parts = model_info.split("|")
            if len(parts) >= 4:
                model_file, mmproj_file, use_api, api_url = parts[0], parts[1], parts[2], parts[3]
            elif len(parts) >= 3:
                model_file, use_api, api_url = parts[0], parts[1], parts[2]
                mmproj_file = ""
        
        # 解析 params_info (格式：max_tokens|ctx_size|temperature|top_p|top_k|repeat_penalty|seed|enable_thinking)
        if params_info and params_info.strip():
            parts = params_info.split("|")
            if len(parts) >= 8:
                max_tokens = int(parts[0])
                ctx_size = int(parts[1])
                temperature = float(parts[2])
                top_p = float(parts[3])
                top_k = int(parts[4])
                repeat_penalty = float(parts[5])
                seed = int(parts[6])
                enable_thinking = parts[7].lower() == "true"
        
        # 解析 video_params_info (格式：max_frames|max_size) - 仅在 video 模式下生效
        if video_params_info and video_params_info.strip():
            parts = video_params_info.split("|")
            if len(parts) >= 2:
                max_frames = int(parts[0])
                max_size = int(parts[1])
                print(f"[llama-yf] Video params: max_frames={max_frames}, max_size={max_size} (only for video mode)")
        
        # 生成最终 prompt
        video = inference_mode == "video"
        if custom_prompt and custom_prompt.strip():
            prompt = PRESET_PROMPTS[preset_prompt].replace("#", custom_prompt.strip()).replace("@", "video" if video else "image")
        else:
            prompt = PRESET_PROMPTS[preset_prompt].replace("@", "video" if video else "image")
        
        use_api_bool = use_api.lower() == "true" if isinstance(use_api, str) else bool(use_api)
        
        has_images = False
        image_list = None
        if image is not None:
            if hasattr(image, '__len__') and len(image) > 0:
                has_images = True
                image_list = image
            elif isinstance(image, torch.Tensor) and image.numel() > 0:
                has_images = True
                image_list = [image]
        
        needs_images = inference_mode in ["images", "video"]
        if needs_images and not has_images:
            raise ValueError(f"Inference mode '{inference_mode}' requires image input")
        
        out1, out2 = "", []
        response, thinking = "", ""
        
        if has_images:
            frames = image_list
            if inference_mode == "one by one":
                tmp_list = []
                for i, img_frame in enumerate(frames):
                    if hasattr(img_frame, 'shape'):
                        h, w = img_frame.shape[0], img_frame.shape[1]
                        if max(h, w) > max_size:
                            scaled_img = self._scale_image(img_frame, max_size)
                            scaled_tensor = torch.from_numpy(scaled_img).unsqueeze(0).float() / 255.0
                        else:
                            scaled_tensor = img_frame.unsqueeze(0).float() / 255.0 if img_frame.dim() == 3 else img_frame.float() / 255.0
                        
                        img_pil = self._tensor_to_image(scaled_tensor.squeeze(0))
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            img_pil.save(tmp.name, "JPEG")
                            image_path = tmp.name
                    
                    if use_api_bool:
                        image_b64_list = []
                        with open(image_path, "rb") as f:
                            image_b64_list.append(base64.b64encode(f.read()).decode("utf-8"))
                        response, thinking = self._invoke_api(
                            api_url=api_url,
                            model_file=model_file,
                            mmproj_file=mmproj_file,
                            ctx_size=ctx_size,
                            prompt=prompt,
                            system_prompt=system_prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repeat_penalty=repeat_penalty,
                            seed=seed,
                            enable_thinking=enable_thinking,
                            image_b64_list=image_b64_list,
                        )
                        print(f"[llama-yf] API mode - response length: {len(response)}, thinking length: {len(thinking)}")
                        # API 模式已经处理了 thinking，不需要再提取
                        if not enable_thinking:
                            thinking = ""
                    else:
                        # CLI 模式需要提取 thinking
                        raw_output = self._invoke_cli(
                            cli_path=cli,
                            model_path=model_path,
                            mmproj_path=mmproj_path,
                            prompt=prompt,
                            system_prompt=system_prompt,
                            image_paths=[image_path] if image_path else None,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repeat_penalty=repeat_penalty,
                            n_gpu_layers=n_gpu_layers,
                            ctx_size=ctx_size,
                            enable_thinking=enable_thinking,
                            seed=seed,
                            threads=threads,
                            disable_warmup=disable_warmup,
                            fit_off=fit_off,
                            force_cpu=force_cpu,
                        )
                        response, thinking = self._extract_thinking(raw_output)
                        print(f"[llama-yf] CLI mode - response length: {len(response)}, thinking length: {len(thinking)}")
                        if not enable_thinking:
                            thinking = ""
                    out2.append(response)
                    if len(frames) > 1:
                        tmp_list.append(f"====== Image {i+1} ======")
                    tmp_list.append(response)
                    
                    if 'image_path' in locals() and os.path.exists(image_path):
                        os.unlink(image_path)
                
                out1 = "\n\n".join(tmp_list)
                print(f"[llama-yf] one by one mode - out1 length: {len(out1)}, out2 items: {len(out2)}")
            else:
                print(f"Processing {len(frames)} images together")
                image_paths = []
                try:
                    for img_frame in frames:
                        scaled_img = self._scale_image(img_frame, max_size)
                        scaled_tensor = torch.from_numpy(scaled_img).unsqueeze(0).float() / 255.0
                        img_pil = self._tensor_to_image(scaled_tensor.squeeze(0))
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            img_pil.save(tmp.name, "JPEG")
                            image_paths.append(tmp.name)
                    
                    if use_api_bool:
                        image_b64_list = []
                        for path in image_paths:
                            with open(path, "rb") as f:
                                image_b64_list.append(base64.b64encode(f.read()).decode("utf-8"))
                        
                        response, thinking = self._invoke_api(
                            api_url=api_url,
                            model_file=model_file,
                            mmproj_file=mmproj_file,
                            ctx_size=ctx_size,
                            prompt=prompt,
                            system_prompt=system_prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repeat_penalty=repeat_penalty,
                            seed=seed,
                            enable_thinking=enable_thinking,
                            image_b64_list=image_b64_list,
                        )
                        print(f"[llama-yf] API mode (multi-image) - response length: {len(response)}, thinking length: {len(thinking)}")
                        # API 模式已经处理了 thinking，不需要再提取
                        if not enable_thinking:
                            thinking = ""
                    else:
                        # CLI 模式需要提取 thinking
                        raw_output = self._invoke_cli(
                            cli_path=cli,
                            model_path=model_path,
                            mmproj_path=mmproj_path,
                            prompt=prompt,
                            system_prompt=system_prompt,
                            image_paths=image_paths if image_paths else None,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repeat_penalty=repeat_penalty,
                            n_gpu_layers=n_gpu_layers,
                            ctx_size=ctx_size,
                            enable_thinking=enable_thinking,
                            seed=seed,
                            threads=threads,
                            disable_warmup=disable_warmup,
                            fit_off=fit_off,
                            max_frames=max_frames,
                            force_cpu=force_cpu,
                        )
                        response, thinking = self._extract_thinking(raw_output)
                        if not enable_thinking:
                            thinking = ""
                    out1 = response
                    out2 = [response]
                finally:
                    for image_path in image_paths:
                        if image_path and os.path.exists(image_path):
                            os.unlink(image_path)
        else:
            print("Text-only inference")
            if use_api_bool:
                response, thinking = self._invoke_api(
                    api_url=api_url,
                    model_file=model_file,
                    mmproj_file=mmproj_file,
                    ctx_size=ctx_size,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    seed=seed,
                    enable_thinking=enable_thinking,
                    image_b64_list=None,  # 纯文本模式
                )
                # API 模式已经处理了 thinking，不需要再提取
                if not enable_thinking:
                    thinking = ""
            else:
                # CLI 模式需要提取 thinking
                raw_output = self._invoke_cli(
                    cli_path=cli,
                    model_path=model_path,
                    mmproj_path=mmproj_path,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    image_paths=None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    n_gpu_layers=n_gpu_layers,
                    ctx_size=ctx_size,
                    enable_thinking=enable_thinking,
                    seed=seed,
                    threads=threads,
                    disable_warmup=disable_warmup,
                    fit_off=fit_off,
                    max_frames=max_frames,
                    force_cpu=force_cpu,
                )
                response, thinking = self._extract_thinking(raw_output)
                if not enable_thinking:
                    thinking = ""
            out1 = response
            out2 = [response]
        
        print(f"Inference success! Response length: {len(out1)} chars")
        print(f"Inference success! Thinking length: {len(thinking)} chars")
        print(f"Inference success! OUTPUT_LIST length: {len(out2)} items")
        output_list_str = "\n\n".join(out2) if out2 else ""
        print(f"Inference success! OUTPUT_LIST string length: {len(output_list_str)} chars")
        return (out1, thinking, output_list_str)

NODE_CLASS_MAPPINGS = {
    "llama-yf_model_select": LlamaModelSelect,
    "llama-yf_params": LlamaParams,
    "llama-yf_video_params": LlamaVideoParams,
    "llama-yf_inference": LlamaInference,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "llama-yf_model_select": "llama-yf 模型选择",
    "llama-yf_params": "llama-yf 参数配置",
    "llama-yf_video_params": "llama-yf 视频参数",
    "llama-yf_inference": "llama-yf 推理",
}