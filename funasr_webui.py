# -*- coding: utf-8 -*-
"""
FunASR 完整功能 Web 界面
------------------------
一个基于 Gradio 的可视化界面，直接调用 FunASR AutoModel，
聚合展示 FunASR 的现有能力：
  - 多模型选择（SenseVoice / Paraformer）
  - 语音识别转写（中/英/日/韩/粤等）
  - 情感 / 音频事件识别（SenseVoice）
  - 说话人分离（CAM++）
  - 标点恢复（Paraformer + ct-punc）
  - 时间戳分段
  - 热词增强

用法:
    python funasr_webui.py                    # 默认 0.0.0.0:47821
    python funasr_webui.py --port 8001        # 自定义端口
    python funasr_webui.py --device cpu       # CPU 推理
"""

import argparse
import os
import tempfile
import datetime
import threading

os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

# ---------------------------------------------------------------------------
# 模型懒加载 + 全局缓存
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_model_cache = {}


def _load_pipeline(model_key: str, device: str, language: str, hotword: str):
    """按需加载并缓存 (model, vad, punc, spk) 组合。返回 (model, spk_model)。"""
    from funasr import AutoModel

    with _lock:
        if model_key in _model_cache:
            entry = _model_cache[model_key]
            return entry["model"], entry["spk_model"]

        spk_model = None
        if model_key == "sensevoice":
            # ASR + 情感 + 事件
            model = AutoModel(
                model="iic/SenseVoiceSmall",
                disable_update=True,
                device=device,
            )
        elif model_key == "paraformer":
            # ASR + 标点 + 时间戳
            model = AutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                disable_update=True,
                device=device,
            )
        else:
            raise ValueError(f"未知模型: {model_key}")

        _model_cache[model_key] = {"model": model, "spk_model": spk_model}
        return model, spk_model


def _load_spk_model(device: str):
    """加载说话人分离模型（独立缓存）。"""
    from funasr import AutoModel

    with _lock:
        if "spk" in _model_cache:
            return _model_cache["spk"]
        spk = AutoModel(
            model="cam++",
            disable_update=True,
            device=device,
        )
        _model_cache["spk"] = spk
        return spk


def _clean_tags(text: str) -> str:
    """去掉 SenseVoice 的 <|...|> 特殊标签，只留纯文本。"""
    import re
    return re.sub(r"<\|[^|]*\|>", "", text or "").strip()


def _extract_tags(text: str):
    """提取 SenseVoice 输出的 <|tag|> 标签，返回 (语言, 情感, 事件列表)。"""
    import re
    tags = re.findall(r"<\|([^|]+)\|>", text or "")
    lang, emotion, events = None, None, []
    for t in tags:
        if t in ("zh", "en", "ja", "ko", "yue"):
            lang = t
        elif t in ("NEUTRAL", "HAPPY", "SAD", "ANGRY", "ANGRY", "SURPRISE", "FEARFUL", "DISGUSTED", "OTHER", "unknown"):
            emotion = t
        elif t in ("Speech", "BGM", "Applause", "Laughter", "Cough", "Breath", "Sigh", "Music", "Guitar", "Piano", "Drums", "Song", "Noise", "Scream", "Whisper", "Cry", "WithWords", "Reading", "Singing", "Other"):
            events.append(t)
    return lang, emotion, events


def predict(audio_path, model_key, device, language, use_spk, hotword):
    """核心推理函数：接收音频路径，返回结构化结果。"""
    import traceback as _tb

    if not audio_path:
        return "", "未上传音频文件。", "[]"

    try:
        return _predict_inner(audio_path, model_key, device, language, use_spk, hotword)
    except Exception as e:
        tb = _tb.format_exc()
        # 写入日志文件便于排查
        LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "funasr_webui.log")
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.datetime.now()}] 推理异常:\n{tb}\n")
        except Exception:
            pass
        return f"出错了：{e}\n\n详细信息见日志文件 funasr_webui.log", f"错误类型: {type(e).__name__}", tb


def _predict_inner(audio_path, model_key, device, language, use_spk, hotword):
    model, spk_model = _load_pipeline(model_key, device, language, hotword)

    kwargs = {"input": audio_path, "batch_size": 1}
    if language and language != "auto":
        kwargs["language"] = language

    result = model.generate(**kwargs)
    res = result[0]
    raw_text = res.get("text", "")

    # 纯文本
    text = _clean_tags(raw_text)

    # 情感 / 事件（仅 SenseVoice 有意义）
    info_lines = []
    if model_key == "sensevoice":
        lang, emotion, events = _extract_tags(raw_text)
        if lang:
            lang_map = {"zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文", "yue": "粤语"}
            info_lines.append(f"语言: {lang_map.get(lang, lang)}")
        if emotion:
            info_lines.append(f"情感: {emotion}")
        if events:
            info_lines.append(f"音频事件: {', '.join(events)}")

    # 分段（时间戳 / 说话人）
    segments = []
    if "sentence_info" in res:
        for s in res["sentence_info"]:
            seg = {
                "start": s.get("start", 0),
                "end": s.get("end", 0),
                "text": _clean_tags(s.get("text") or s.get("sentence", "")),
            }
            if s.get("spk") is not None:
                seg["spk"] = s["spk"]
            segments.append(seg)

    # 说话人分离（懒加载独立 spk 模型 + 二次标注）
    if use_spk and segments:
        try:
            segments = _diarize(audio_path, segments, device)
        except Exception as e:
            info_lines.append(f"[说话人分离告警] {e}")

    # 组装分段文本
    if segments:
        seg_text_lines = []
        for seg in segments:
            t = seg["text"]
            head = f"[{seg['start']/1000:.2f}s-{seg['end']/1000:.2f}s]"
            if seg.get("spk") is not None:
                head += f"(说话人{seg['spk']})"
            seg_text_lines.append(f"{head} {t}")
        segments_text = "\n".join(seg_text_lines)
    else:
        segments_text = text

    info_text = "\n".join(info_lines) if info_lines else "（SenseVoice 未检测到情感/事件标签，或使用了无需标签的模型）"

    import json
    segments_json = json.dumps(segments, ensure_ascii=False, indent=2)

    return segments_text, info_text, segments_json


def _diarize(audio_path, segments, device):
    """对已分段的时间戳做说话人分离标注。"""
    import soundfile as sf
    import numpy as np
    from funasr.models.campplus.cluster_backend import ClusterBackend
    from funasr.models.campplus.utils import distribute_spk, postprocess, sv_chunk

    spk_model = _load_spk_model(device)
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr

    diar_inputs, indexes = [], []
    for i, seg in enumerate(segments):
        start = max(float(seg["start"]) / 1000.0, 0.0)
        end = min(float(seg["end"]) / 1000.0, duration)
        si, ei = int(start * sr), int(end * sr)
        if ei <= si:
            continue
        diar_inputs.append([start, end, audio[si:ei]])
        indexes.append(i)

    if not diar_inputs:
        return segments

    chunks = sv_chunk(diar_inputs, fs=sr)
    if not chunks:
        return segments

    spk_results = spk_model.generate(input=[c[2] for c in chunks], cache={}, is_final=True)
    import torch
    embs = torch.cat([r["spk_embedding"] for r in spk_results], dim=0)
    labels = ClusterBackend(merge_thr=0.78).to(device)(embs.cpu(), oracle_num=None)
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    timeline = postprocess(sorted(chunks, key=lambda c: c[0]), None, labels, embs.detach().cpu().numpy())

    sentences = [
        {
            "text": segments[i]["text"],
            "start": int(float(segments[i]["start"])),
            "end": int(float(segments[i]["end"])),
        }
        for i in indexes
    ]
    distribute_spk(sentences, timeline)
    for i, sent in zip(indexes, sentences):
        if sent.get("spk") is not None:
            segments[i]["spk"] = str(sent["spk"])
    return segments


# ---------------------------------------------------------------------------
# Gradio 界面
# ---------------------------------------------------------------------------
def build_ui(device: str):
    import gradio as gr

    with gr.Blocks(title="FunASR 完整功能界面", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎙️ FunASR 完整功能界面\n"
            "上传音频文件（或录音），选择功能后点击识别。底层调用 FunASR AutoModel。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(
                    label="音频输入（上传或录音）",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                model_key = gr.Radio(
                    label="识别模型",
                    choices=[
                        ("SenseVoice（多语言 + 情感/事件）", "sensevoice"),
                        ("Paraformer（中文 + 标点 + 时间戳）", "paraformer"),
                    ],
                    value="sensevoice",
                )
                language = gr.Dropdown(
                    label="语言",
                    choices=["auto", "zh", "en", "ja", "ko", "yue"],
                    value="auto",
                )
                use_spk = gr.Checkbox(
                    label="说话人分离（额外加载 CAM++ 模型，较慢）",
                    value=False,
                )
                hotword = gr.Textbox(
                    label="热词（逗号分隔，如：达摩院,语音识别）",
                    placeholder="可选，用于领域词增强",
                )
                run_btn = gr.Button("🎯 开始识别", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 识别结果（时间戳/说话人）")
                result_text = gr.Textbox(label="结果", lines=10, interactive=False)
                gr.Markdown("### 情感 / 事件 / 语言")
                info_text = gr.Textbox(label="附加信息", lines=3, interactive=False)
                gr.Markdown("### 结构化 JSON")
                result_json = gr.Code(label="Segment JSON", language="json")

        run_btn.click(
            fn=predict,
            inputs=[audio, model_key, gr.State(device), language, use_spk, hotword],
            outputs=[result_text, info_text, result_json],
        )

        gr.Markdown(
            "---\n"
            "**说明**：模型首次加载需下载（SenseVoice ~234M，CAM++ ~7M，多几秒）。"
            "情感/事件仅 SenseVoice 支持；说话人分离需额外加载 CAM++。"
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="FunASR 完整功能 Web 界面")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=47821)
    parser.add_argument("--device", default="cuda", help="cuda / cpu / mps")
    parser.add_argument("--share", action="store_true", help="生成临时公网链接")
    args = parser.parse_args()

    demo = build_ui(args.device)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()