# 运行命令

- 项目：FunASR（ModelScope 语音识别工具箱）
- 生成时间：2026-09-09
- 运行方式：直接运行（Python / `funasr-server`）；可选 Docker；CPU/边缘可用 llama.cpp GGUF
- 硬件评估：**满足**（空卡 RTX 4080 16GB）
  - 依据：官方 Quickstart 默认 `device="cuda"` 加载 SenseVoiceSmall / Paraformer 等工业小模型；`docs/vllm_guide_zh.md` 写明 GPU ≥ 8GB VRAM、推荐 16GB+，本机 16GB 落在推荐区间。非常规大 LLM 路径另议，默认/常规精度可空卡跑通。

## 环境准备

本机已有 `.venv` 与 `funasr-server`。若需重装：

```powershell
# 建议：uv 建 venv 后装匹配驱动的 PyTorch CUDA wheel，再装 funasr
# ffmpeg（本机已有）：E:\Programs\ffmpeg-master-latest-win64-gpl\bin
$env:Path = "E:\Programs\ffmpeg-master-latest-win64-gpl\bin;" + $env:Path
$env:HF_ENDPOINT = 'https://hf-mirror.com'
```

官方要点：先确认 `torch.cuda.is_available()` 为 True，再使用 `device="cuda"`。

## 启动

- 推荐：`pwsh -NoProfile -File .\start.ps1`
- 说明：启动后按提示选择服务；**不要默认全开**。默认选项为 **webui**（本机 `funasr_webui.py`）。
- 服务菜单：`webui`（Gradio，端口起点 47821）/ `server`（`funasr-server`，端口起点 8000）/ `sensevoice`（官方 demo CLI，无端口）
- 自动化：`-Service webui|server|sensevoice`

等价手动命令：

```powershell
$env:Path = "E:\Programs\ffmpeg-master-latest-win64-gpl\bin;" + $env:Path
# webui
.\.venv\Scripts\python.exe .\funasr_webui.py --host 127.0.0.1 --port 47821 --device cuda
# server（OpenAI 兼容 API）
.\.venv\Scripts\funasr-server.exe --model sensevoice --device cuda --port 8000
# sensevoice CLI demo
.\.venv\Scripts\python.exe .\examples\industrial_data_pretraining\sense_voice\demo.py
```

## 验证

- `python -c "import torch; print(torch.cuda.is_available())"` → True
- 对样例 wav/mp3 调用 `AutoModel.generate` 有文本输出
- 或访问 `funasr-server` 健康/转写接口无致命报错

## 备注

- 分类建议：**Pro2（满足）**
- 本机基准：空卡视角 16GB；忽略当前占卡
- llama.cpp / GGUF / Vulkan 为可选 CPU·端侧路径，非默认必需
- 勿在未评估前提下默认拉起 vLLM 多卡或超大模型
