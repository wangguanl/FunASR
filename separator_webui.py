# -*- coding: utf-8 -*-
"""
音频分离界面（自包含版）
------------------------
一个文件搞定：Gradio 界面 + 24bit 修复 + ffmpeg 环境 + 模型检查下载 + 自动开浏览器。

用法（二选一）：
  1. 项目根目录执行 .\start.ps1
  2. 命令行：& "e:\Pro2\audio-demucs\python-audio-separator\.venv\Scripts\python.exe" separator_webui.py
"""

import os
import sys
import time
import socket
import logging
import tempfile
import threading
import webbrowser
import subprocess
import urllib.request

import gradio as gr
from audio_separator.separator import Separator

# ---------------------------------------------------------------------------
# 路径配置（绝对路径，摆脱对运行目录的依赖）
# ---------------------------------------------------------------------------
MODEL_DIR = r"c:\Users\wang\audio-separator-models"
FFMPEG_BIN = r"e:\Pro2\audio-demucs\python-audio-separator\ffmpeg\bin"
FFMPEG = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
MODEL_FILE = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MODEL_URL = ("https://gh-proxy.com/https://github.com/TRvlvr/model_repo/releases/"
             "download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt")

HOST = "127.0.0.1"
PREFERRED_PORT = 7860

DEFAULT_MODEL = MODEL_FILE
OUTPUT_FORMATS = ["WAV", "FLAC", "MP3", "OGG", "M4A", "ALAC"]
STEM_CHOICES = ["全部音轨", "Vocals", "Instrumental", "Drums", "Bass", "Other", "Guitar", "Piano"]
MAX_PREVIEWS = 8
BROWSER_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".m4a", ".opus"}


# ---------------------------------------------------------------------------
# 环境准备
# ---------------------------------------------------------------------------
def ensure_ffmpeg():
    """把项目自带 ffmpeg 加入 PATH，供分离库调用。"""
    if os.path.isdir(FFMPEG_BIN) and FFMPEG_BIN not in os.environ.get("PATH", ""):
        os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")


def ensure_model():
    """确保完整模型存在，缺失则经加速镜像下载。"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    dest = os.path.join(MODEL_DIR, MODEL_FILE)
    if os.path.exists(dest) and os.path.getsize(dest) > 500 * 1024 * 1024:
        print(f"== 模型已就绪: {dest}")
        return
    print("== 模型缺失或不完整，开始下载（约 610MB，请耐心等待）...")
    tmp = dest + ".part"
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done, last = 0, time.time()
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last >= 2:
                print(f"\r   下载中: {done/1e6:.1f}/{total/1e6:.1f} MB ({100*done/total:.1f}%)", flush=True)
                last = time.time()
    print()
    os.replace(tmp, dest)
    size = os.path.getsize(dest)
    if size < 500 * 1024 * 1024:
        raise RuntimeError(f"下载仍不完整: {size/1e6:.1f}MB，请重试。")
    print(f"== 模型下载完成: {dest}")


def find_free_port(start: int = PREFERRED_PORT, host: str = HOST, max_tries: int = 50) -> int:
    """从 start 起找第一个可绑定端口；占用就顺延，不假定是本界面。"""
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"从 {start} 起连续 {max_tries} 个端口都不可用")


# ---------------------------------------------------------------------------
# 分离逻辑
# ---------------------------------------------------------------------------
class _LastErrorHandler(logging.Handler):
    """捕获 Separator 内部吞掉的 ERROR 日志，便于把真实错误显示到界面。"""

    def __init__(self):
        super().__init__()
        self.last_error = None

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            msg = record.getMessage()
            if record.exc_info and record.exc_info[1]:
                msg = f"{msg}: {record.exc_info[1]}"
            self.last_error = msg


def _build_model_choices():
    try:
        sep = Separator(model_file_dir=MODEL_DIR, info_only=True)
        simplified = sep.list_supported_model_files()
        files = sorted(f for f in simplified.keys())
        return files if DEFAULT_MODEL in files else [DEFAULT_MODEL] + files
    except Exception as e:  # noqa: BLE001
        print(f"加载模型列表失败，使用默认模型: {e}")
        return [DEFAULT_MODEL]


def _to_wav16(src_path):
    """用 ffmpeg 把任意输入统一转成 16bit/44.1kHz 的双声道 WAV。

    修复：Windows 上 libsndfile 读取 24bit FLAC 等格式会抛
    `LibsndfileError: Internal psf_fseek() failed`，导致分离返回空结果。
    先转成 16bit WAV 即可稳定绕过。返回 wav 路径。
    """
    fd, wav = tempfile.mkstemp(suffix=".wav", prefix="sep_in16_")
    os.close(fd)
    cmd = [FFMPEG, "-y", "-i", src_path, "-ac", "2", "-sample_fmt", "s16", "-ar", "44100", wav]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(wav) or os.path.getsize(wav) == 0:
        tail = (proc.stderr or proc.stdout or "")[-500:]
        raise gr.Error(f"音频解码失败，无法读取该文件。\n{tail}")
    return wav


def _preview_audio_path(src_path: str) -> str:
    """浏览器播不了的格式（FLAC/ALAC 等）先转成 WAV，供页面试听。"""
    ext = os.path.splitext(src_path)[1].lower()
    if ext in BROWSER_AUDIO_EXTS:
        return src_path
    return _to_wav16(src_path)


def _preview_updates(output_files):
    updates = []
    for i in range(MAX_PREVIEWS):
        if i < len(output_files):
            path = output_files[i]
            label = os.path.splitext(os.path.basename(path))[0]
            updates.append(
                gr.Audio(
                    value=_preview_audio_path(path),
                    label=label,
                    visible=True,
                )
            )
        else:
            updates.append(gr.Audio(value=None, label=f"音轨 {i + 1}", visible=False))
    return updates


def separate_audio(
    input_file,
    model_filename,
    output_format,
    single_stem,
    chunk_duration,
    progress=gr.Progress(track_tqdm=True),
):
    """执行分离并返回输出文件列表。"""
    if input_file is None:
        raise gr.Error("请先上传一个音频文件。")

    src_path = input_file
    if not os.path.exists(src_path):
        raise gr.Error(f"找不到上传的文件: {src_path}")

    single_stem_val = None if single_stem == "全部音轨" else single_stem

    progress(0.03, desc="预处理音频（转 16bit WAV）…")
    wav_path = _to_wav16(src_path)

    out_dir = tempfile.mkdtemp(prefix="audio_sep_out_")

    try:
        progress(0.1, desc="初始化分离器…")
        separator = Separator(
            model_file_dir=MODEL_DIR,
            output_dir=out_dir,
            output_format=output_format,
            output_single_stem=single_stem_val,
            chunk_duration=chunk_duration if chunk_duration and chunk_duration > 0 else None,
        )

        progress(0.2, desc=f"加载模型 {model_filename}…")
        separator.load_model(model_filename=model_filename)

        progress(0.3, desc="开始分离…")

        handler = _LastErrorHandler()
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            output_files = separator.separate(wav_path)
        finally:
            root_logger.removeHandler(handler)

        if not output_files:
            detail = handler.last_error or "未捕获到具体错误，请检查音频文件是否为常见格式（wav/mp3/flac/m4a）。"
            raise gr.Error(f"分离未产生任何输出文件。\n{detail}")

        output_files = [f if os.path.isabs(f) else os.path.join(out_dir, f) for f in output_files]

        progress(1.0, desc="完成")
        status = "✅ 分离完成，可在右侧试听或下载。"
        return [output_files, *_preview_updates(output_files), status]
    finally:
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Gradio 界面
# ---------------------------------------------------------------------------
def build_ui():
    model_choices = _build_model_choices()

    with gr.Blocks(title="Audio Separator Web UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎵 Audio Separator Web UI\n"
            "基于 `python-audio-separator` 的本地音频分离界面。上传音频，选择模型，即可分离人声与伴奏。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="上传音频文件",
                    file_types=["audio", ".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".ac3"],
                )
                model = gr.Dropdown(
                    choices=model_choices,
                    value=DEFAULT_MODEL if DEFAULT_MODEL in model_choices else model_choices[0],
                    label="分离模型",
                )
                fmt = gr.Dropdown(choices=OUTPUT_FORMATS, value="FLAC", label="输出格式")
                stem = gr.Dropdown(choices=STEM_CHOICES, value="全部音轨", label="输出音轨")
                chunk = gr.Slider(
                    minimum=0,
                    maximum=1200,
                    step=60,
                    value=0,
                    label="分块时长（秒，0=不分块；长音频建议 300-600）",
                )
                run_btn = gr.Button("开始分离", variant="primary", size="lg")

            with gr.Column(scale=1):
                status = gr.Markdown("等待上传并开始分离…")
                gr.Markdown("### 在线试听")
                preview_players = [
                    gr.Audio(
                        label=f"音轨 {i + 1}",
                        type="filepath",
                        interactive=False,
                        visible=False,
                        show_download_button=True,
                    )
                    for i in range(MAX_PREVIEWS)
                ]
                outputs = gr.Files(label="分离结果（可下载）")

        run_btn.click(
            separate_audio,
            inputs=[input_file, model, fmt, stem, chunk],
            outputs=[outputs, *preview_players, status],
        )

        gr.Markdown(
            "> 提示：模型缓存于 `%s`。上传后会先统一转为 16bit WAV 再分离，兼容 24bit 等高比特率音频。" % MODEL_DIR
        )

    return demo


def main():
    ensure_ffmpeg()
    print("== ffmpeg:", os.path.exists(FFMPEG))

    port = find_free_port()
    url = f"http://{HOST}:{port}"
    if port != PREFERRED_PORT:
        print(f"== 端口 {PREFERRED_PORT} 已被占用，改用 {port}")

    ensure_model()

    print(f"== 启动界面: {url}")
    threading.Thread(target=lambda: (time.sleep(2), webbrowser.open(url)), daemon=True).start()
    app = build_ui()
    app.launch(server_name=HOST, server_port=port, show_error=True, prevent_thread_lock=False)


if __name__ == "__main__":
    main()