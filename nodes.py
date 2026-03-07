# ComfyUI-Qwen3.5 GGUF Plugin
# Fast inference node using llama.cpp (via llama-mtmd-cli subprocess)
# Model files are loaded from ComfyUI's models/LLM directory

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image

import folder_paths

THINK_BLOCK_RE = re.compile(
    r"<think[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL
)


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


class Qwen35GGUF:
    """Qwen3.5 GGUF node — fast inference via llama.cpp."""

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
                "prompt": ("STRING", {
                    "default": "详细描述这张图片。",
                    "multiline": True,
                    "tooltip": "Text prompt for the model (支持中文)",
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional system prompt to set model behavior (支持中文)",
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
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Image for vision tasks"}),
                "cli_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to llama-mtmd-cli binary. Auto-detected if empty.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "THINKING")
    FUNCTION = "process"
    CATEGORY = "Qwen3.5"

    @classmethod
    def IS_CHANGED(cls, model_file, mmproj_file, *args, **kwargs):
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
                f"[Qwen3.5 GGUF] Model file not found: {model_path_obj}\n"
                f"Available GGUF files in {llm_dir}:\n  {file_list}"
            )
        
        if not mmproj_path_obj.exists():
            files = [str(f.relative_to(llm_dir)) for f in llm_dir.rglob("*.gguf") if "mmproj" in f.name.lower()]
            file_list = "\n  ".join(files)
            raise FileNotFoundError(
                f"[Qwen3.5 GGUF] MMProj file not found: {mmproj_path_obj}\n"
                f"Available mmproj files in {llm_dir}:\n  {file_list if file_list else '-- 无 --'}"
            )
        
        print(f"[Qwen3.5 GGUF] Using model: {model_path_obj}")
        print(f"[Qwen3.5 GGUF] Using mmproj: {mmproj_path_obj}")
        
        return model_path_obj, mmproj_path_obj

    @staticmethod
    def _find_cli(cli_path_override: str) -> str:
        """Find the llama-mtmd-cli binary - 优先使用插件目录内的版本"""
        
        # 1. 优先使用用户通过参数指定的路径
        if cli_path_override and cli_path_override.strip():
            p = Path(cli_path_override.strip())
            if p.is_file() and os.access(str(p), os.X_OK):
                print(f"[Qwen3.5 GGUF] Using specified CLI: {p}")
                return str(p)
            else:
                print(f"[Qwen3.5 GGUF] Warning: Specified CLI not found: {p}")
        
        # 2. 使用插件目录内的llama-mtmd-cli.exe
        plugin_dir = Path(__file__).parent
        plugin_cli = plugin_dir / "llama" / "llama-mtmd-cli.exe"
        if plugin_cli.is_file() and os.access(str(plugin_cli), os.X_OK):
            print(f"[Qwen3.5 GGUF] Using plugin CLI: {plugin_cli}")
            return str(plugin_cli)
        
        # 3. 检查当前目录的llama子目录
        local_llama = Path.cwd() / "llama" / "llama-mtmd-cli.exe"
        if local_llama.is_file() and os.access(str(local_llama), os.X_OK):
            print(f"[Qwen3.5 GGUF] Using local llama CLI: {local_llama}")
            return str(local_llama)
        
        # 4. 最后才去PATH找
        found = shutil.which("llama-mtmd-cli")
        if found:
            print(f"[Qwen3.5 GGUF] ⚠️  Using PATH CLI: {found}")
            return found
        
        raise FileNotFoundError(
            "[Qwen3.5 GGUF] llama-mtmd-cli not found. "
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
        image_path: str | None,
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
            "-ngl", str(n_gpu_layers),
            "-c", str(ctx_size),
            "--seed", str(actual_seed),
            "-t", str(threads),
        ]

        # 如果有图像，添加图像参数
        if image_path is not None:
            abs_image_path = os.path.abspath(image_path)
            cmd.extend(["--image", abs_image_path])
            print(f"[Qwen3.5 GGUF] Using image: {abs_image_path}")

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
        print(f"[Qwen3.5 GGUF] Running command: {cmd[0]} ...")
        print(f"[Qwen3.5 GGUF] === 生成参数 ===")
        print(f"[Qwen3.5 GGUF]   - 模型: {model_path.name}")
        print(f"[Qwen3.5 GGUF]   - 视觉模型: {mmproj_path.name}")
        print(f"[Qwen3.5 GGUF]   - 最大 tokens: {max_tokens}")
        print(f"[Qwen3.5 GGUF]   - 温度: {temperature}")
        print(f"[Qwen3.5 GGUF]   - Top-P: {top_p}")
        print(f"[Qwen3.5 GGUF]   - Top-K: {top_k}")
        print(f"[Qwen3.5 GGUF]   - 重复惩罚: {repeat_penalty}")
        print(f"[Qwen3.5 GGUF]   - GPU层数: {n_gpu_layers}")
        print(f"[Qwen3.5 GGUF]   - 上下文: {ctx_size}")
        print(f"[Qwen3.5 GGUF]   - 线程数: {threads}")
        print(f"[Qwen3.5 GGUF]   - 随机种子: {actual_seed}" if seed != -1 else f"[Qwen3.5 GGUF]   - 随机种子: {actual_seed} (random)")
        if image_path is not None:
            print(f"[Qwen3.5 GGUF]   - 图像: {Path(image_path).name}")
        print(f"[Qwen3.5 GGUF]   - 提示词: {prompt[:50]}..." if len(prompt) > 50 else f"[Qwen3.5 GGUF]   - 提示词: {prompt}")
        print(f"[Qwen3.5 GGUF] ===================")
        
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

            print(f"[Qwen3.5 GGUF] Return code: {result.returncode}")
            
            # 处理 stderr，只显示关键状态信息
            if result.stderr:
                # 只显示关键的 llama.cpp 状态信息
                for line in result.stderr.split('\n'):
                    if 'image slice encoded' in line or 'image decoded' in line or 'decoding image' in line:
                        if 'image slice encoded' in line:
                            print(f"[Qwen3.5 GGUF] ✅ 图像编码")
                        elif 'decoding image' in line:
                            print(f"[Qwen3.5 GGUF] ✅ 图像解码")
                        elif 'image decoded' in line:
                            print(f"[Qwen3.5 GGUF] ✅ 图像处理完成")

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                print(f"[Qwen3.5 GGUF] ❌ 错误输出:")
                print(f"{error_msg}")
                raise RuntimeError(
                    f"[Qwen3.5 GGUF] Inference failed (exit {result.returncode}): {error_msg}"
                )

            return result.stdout

        except subprocess.TimeoutExpired:
            raise RuntimeError("[Qwen3.5 GGUF] Inference timed out after 300 seconds")
        except Exception as e:
            raise RuntimeError(f"[Qwen3.5 GGUF] Failed to run inference: {str(e)}")

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
        prompt: str,
        system_prompt: str,
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
        image=None,
        cli_path: str = "",
    ):
        """处理输入并返回模型输出"""
        
        print(f"[Qwen3.5 GGUF] Starting inference with {model_file}")
        
        # 查找 CLI 路径
        cli = Qwen35GGUF._find_cli(cli_path)
        
        # 确保模型文件存在
        model_path, mmproj_path = Qwen35GGUF._ensure_model(model_file, mmproj_file)

        image_path = None
        try:
            # 处理图像输入
            if image is not None:
                image_path = Qwen35GGUF._tensor_to_temp_image(image)
                print(f"[Qwen3.5 GGUF] Saved temporary image: {image_path}")

            # 调用 CLI 进行推理
            raw_output = Qwen35GGUF._invoke_cli(
                cli_path=cli,
                model_path=model_path,
                mmproj_path=mmproj_path,
                prompt=prompt,
                system_prompt=system_prompt,
                image_path=image_path,
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
            )

            # 提取思考内容和响应
            response, thinking = Qwen35GGUF._extract_thinking(raw_output)

            if not enable_thinking:
                thinking = ""

            print(f"[Qwen3.5 GGUF] Success! Response length: {len(response)} chars")
            
            return (response, thinking)

        finally:
            # 清理临时文件
            if image_path and os.path.exists(image_path):
                try:
                    os.unlink(image_path)
                    print(f"[Qwen3.5 GGUF] Cleaned up temp file: {image_path}")
                except Exception as e:
                    print(f"[Qwen3.5 GGUF] Warning: Could not delete temp file {image_path}: {e}")


NODE_CLASS_MAPPINGS = {"Qwen35GGUF": Qwen35GGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen35GGUF": "Qwen 3.5 (GGUF)"}