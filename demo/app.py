#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py - Gradio demo for TiTok multi-segment audio-visual temporal grounding.

Upload a video (or give a YouTube URL) and a question; the answer is shown as text and
the predicted segments on a timeline. Inference reuses the GDPO trainer's model loader
(load_model_and_tokenizer) and LazySupervisedDataset, i.e. the same preprocessing as training.

Usage (GPU machine, from the repo root):
  pip install gradio
  bash demo/run_demo.sh            # TiTok GDPO checkpoint-2000 on top of the SFT model

env:
  DEMO_MODEL_PATH  [required] LoRA adapter or merged model (same as gdpo --model_path)
  DEMO_MODEL_BASE  [required] base model (same as gdpo --model_base)
  DEMO_TTI_MODE    off|on (default on). Must match the mode used in training.
  DEMO_SHARE       1|0 (default 0). 1 creates a temporary public gradio.live link.
  DEMO_SERVER_NAME default 127.0.0.1
  DEMO_SERVER_PORT default 7860

Note: written without GPU access; the generate call and preprocessing are marked [VERIFY]
      below in case they need adjusting for your environment.
"""

import os
import sys
import re
import json
import shutil
import tempfile
import subprocess
from dataclasses import dataclass

import torch

# -- path setup --
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "titok"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "gdpo"))

# Reuse the GDPO trainer's model loader so the demo loads models exactly as training does.
from gdpo_trainer import load_model_and_tokenizer

from qwenvl.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from transformers import WhisperFeatureExtractor

import gradio as gr

# -- workaround for a gradio_client api_info bug --
# Some gradio versions cannot handle boolean schemas (additionalProperties: true) when
# building api_info and fail with "argument of type 'bool' is not iterable".
# Treat any non-dict schema as 'Any' in get_type / _json_schema_to_python_type.
try:
    import gradio_client.utils as _gcu

    _orig_get_type = _gcu.get_type
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)
    _gcu.get_type = _safe_get_type

    _orig_js = _gcu._json_schema_to_python_type
    def _safe_js(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_js(schema, defs)
    _gcu._json_schema_to_python_type = _safe_js
except Exception as _e:
    print(f"[DEMO] gradio_client api_info patch skip: {_e}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Settings (env)
# ============================================================
MODEL_PATH = os.environ.get("DEMO_MODEL_PATH")
MODEL_BASE = os.environ.get("DEMO_MODEL_BASE")
TTI_MODE = os.environ.get("DEMO_TTI_MODE", "on")
SHARE = os.environ.get("DEMO_SHARE", "0") not in ("0", "false", "False")
SERVER_NAME = os.environ.get("DEMO_SERVER_NAME", "127.0.0.1")
SERVER_PORT = int(os.environ.get("DEMO_SERVER_PORT", "7860"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not MODEL_PATH or not MODEL_BASE:
    raise SystemExit("[DEMO] DEMO_MODEL_PATH and DEMO_MODEL_BASE must be set")


# ============================================================
# Time-token decoding (same as eval/eval_miou.py)
# ============================================================
def decode_vtg_time(token_str, max_time=9999.9):
    has_dot = "<tdot>" in token_str
    if has_dot:
        parts = token_str.split("<tdot>")
        int_part = re.findall(r"<t(\d)>", parts[0])
        dec_part = re.findall(r"<t(\d)>", parts[1]) if len(parts) > 1 else []
    else:
        int_part = re.findall(r"<t(\d)>", token_str)
        dec_part = []
    if not int_part:
        return None
    integer_part = int("".join(int_part))
    decimal_part = int(dec_part[0]) if dec_part else 0
    t = integer_part + decimal_part / 10.0
    return min(t, max_time)


def parse_multi_segments(raw, max_time=9999.9):
    """'From X to Y. From X to Y.' -> [[start, end], ...] (seconds)."""
    segments = []
    pattern = r"[Ff]rom\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)\s+to\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)"
    for m in re.finditer(pattern, raw):
        start = decode_vtg_time(m.group(1), max_time)
        end = decode_vtg_time(m.group(2), max_time)
        if start is not None and end is not None:
            if end <= start:
                end = min(start + 0.1, max_time)
            segments.append([start, end])
    return segments


def clean_text(raw):
    """Strip special tokens for display."""
    txt = raw
    for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        txt = txt.replace(tok, "")
    return txt.strip()


# ============================================================
# Answer-format instruction used in training/eval (same as the human prompts in data/)
#   Without it the model does not answer in the <t..> time-token format.
# ============================================================
ANSWER_FORMAT_BLOCK = (
    '\n\nAnswer format:\n'
    '"From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>."\n\n'
    'For multiple segments, Separate multiple segments with a period and space, like:\n'
    '"From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>. '
    'From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>. ..."'
)


def build_human_prompt(question):
    """Append the training Answer-format block to the user question.

    If the question already contains 'Answer format' (a full prompt), it is kept as is.
    The <image> placeholder is replaced with <video> by the dataset when a video is given.
    """
    q = (question or "").strip()
    if "answer format" not in q.lower():
        q = q + ANSWER_FORMAT_BLOCK
    return "<image>\n" + q


def _resolve_ffmpeg():
    """Locate ffmpeg: system PATH first, then the imageio-ffmpeg bundled binary.

    Without a system ffmpeg, `pip install imageio-ffmpeg` provides a static binary
    (no root needed).
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def download_youtube_video(url):
    """Download a YouTube (or other yt-dlp supported) URL to a temporary mp4.

    Returns a local mp4 path so it follows the same path as an upload.
    Returns None on failure. The caller must delete the returned file.
    """
    try:
        import yt_dlp
    except ImportError:
        print("[DEMO] yt-dlp is not installed; cannot download URLs (`pip install yt-dlp`)")
        return None

    out_tmpl = tempfile.NamedTemporaryFile(suffix=".%(ext)s", delete=False).name
    # Prefer mp4 up to 720p (fast, small). With ffmpeg, best video+audio can also be merged.
    ydl_opts = {
        "format": "best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg:
        ydl_opts["ffmpeg_location"] = os.path.dirname(ffmpeg)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)
    except Exception as e:
        print(f"[DEMO] YouTube download failed: {repr(e)[:200]}")
        return None
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        print("[DEMO] downloaded file is empty")
        return None
    print(f"[DEMO] YouTube download finished: {path}")
    return path


def extract_audio_from_video(video_path):
    """Extract the audio track of a video as 16 kHz mono wav when no audio is uploaded.

    Returns a temporary wav path so it follows the same path as an uploaded audio file.
    Returns None without ffmpeg or when there is no audio track (video-only inference).
    The caller must delete the returned file.
    """
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        print("[DEMO] ffmpeg not found; skipping audio extraction (video only). "
              "For audio, `pip install imageio-ffmpeg` or "
              "`conda install -c conda-forge ffmpeg`")
        return None
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", video_path, "-vn",
             "-ac", "1", "-ar", str(DemoDataArgs.sampling_rate), "-f", "wav", out],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        tail = e.stderr.decode(errors="ignore")[-200:] if e.stderr else ""
        print(f"[DEMO] audio extraction failed (silent / no track?); using video only: {tail}")
        try:
            os.remove(out)
        except OSError:
            pass
        return None
    # guard against empty files (silent track)
    if not os.path.exists(out) or os.path.getsize(out) < 1024:
        print("[DEMO] extracted audio is empty; using video only")
        try:
            os.remove(out)
        except OSError:
            pass
        return None
    print(f"[DEMO] extracted audio from video: {out}")
    return out


# ============================================================
# Data preprocessing args (copy of GDPODataArgs in gdpo/gdpo_trainer.py main())
#   Values are kept identical so that preprocessing matches training.
# ============================================================
@dataclass
class DemoDataArgs:
    dataset_use: str = ""
    model_type: str = "qwen2.5vl"
    video_max_frames: int = 128
    video_min_frames: int = 64
    base_interval: float = 0.2
    max_pixels: int = 176400
    min_pixels: int = 784
    video_max_frame_pixels: int = 28224
    video_min_frame_pixels: int = 784
    video_max_total_pixels: int = 1664 * 28 * 28
    video_min_total_pixels: int = 256 * 28 * 28
    run_test: bool = False
    do_sample: bool = False
    num_sample: int = 1
    train_type: str = "sft"
    tti_time_format: str = "special_token"
    feature_size: int = 128
    chunk_length: int = 30
    hop_length: int = 160
    sampling_rate: int = 16000
    image_processor: object = None
    audio_processor: object = None


# ============================================================
# Load the model (once)
# ============================================================
print(f"[DEMO] loading model (tti_mode={TTI_MODE}, device={DEVICE})")
model, tokenizer = load_model_and_tokenizer(MODEL_PATH, MODEL_BASE, tti_mode=TTI_MODE)
model = model.to(DEVICE).eval()
# Generate with the base under the PEFT wrapper (as in GDPO; LoRA is applied inline)
GEN_MODEL = model.get_base_model() if hasattr(model, "get_base_model") else model

data_args = DemoDataArgs()
data_args.tti_time_format = "special_token" if TTI_MODE == "on" else "off"
data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(MODEL_BASE)
data_args.audio_processor = WhisperFeatureExtractor(
    feature_size=data_args.feature_size,
    sampling_rate=data_args.sampling_rate,
    hop_length=data_args.hop_length,
    chunk_length=data_args.chunk_length,
)
collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
_VIDEO_TOKEN_ID = 151656
print("[DEMO] model ready")


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def infer(video_path, audio_path, question, max_new_tokens, do_sample, temperature,
          youtube_url=None):
    # Without an uploaded video, download the URL (then handled like an upload)
    downloaded_video = None
    if not video_path and youtube_url and youtube_url.strip():
        downloaded_video = download_youtube_video(youtube_url.strip())
        video_path = downloaded_video
    if not video_path:
        return "[Upload a video or enter a YouTube URL]", "", []
    if not question or not question.strip():
        question = ("At what point in the video does the described event occur "
                    "in terms of both video and audio?")

    # 1) Build a single-sample JSON (LazySupervisedDataset schema)
    #    with the same Answer-format block as training/eval (build_human_prompt).
    sample = {
        "video": video_path,
        "conversations": [
            {"from": "human", "value": build_human_prompt(question)},
            {"from": "gpt", "value": ""},
        ],
    }
    # Without uploaded audio, extract it from the video (then handled like an upload)
    extracted_audio = None
    if not audio_path and video_path:
        extracted_audio = extract_audio_from_video(video_path)
        audio_path = extracted_audio
    if audio_path:
        sample["audio"] = audio_path

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump([sample], f, ensure_ascii=False)
        tmp_json = f.name

    try:
        data_args.dataset_use = tmp_json
        ds = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
        item = ds[0]
        batch = collator([item])
    finally:
        for _tmp in (tmp_json, extracted_audio, downloaded_video):
            if _tmp:
                try:
                    os.remove(_tmp)
                except OSError:
                    pass

    device = DEVICE
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # 2) Keep only the prompt (as in GDPO compute_loss: empty gpt answer -> cut after the assistant prompt)
    labels = batch.get("labels", None)
    if labels is not None:
        labels = labels.to(device)
        ans = (labels[0] != -100).nonzero(as_tuple=True)[0]
        if len(ans) > 0:
            cut = ans[0].item()
            input_ids = input_ids[:, :cut]
            attention_mask = attention_mask[:, :cut]

    # 3) Multimodal kwargs (same keys as the GDPO gen_kwargs)
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=1.0, top_k=50,
                          repetition_penalty=1.0)
    else:
        gen_kwargs.update(do_sample=False)

    pvv = batch.get("pixel_values_videos", None)
    if pvv is not None:
        gen_kwargs["pixel_values_videos"] = pvv.to(device=device, dtype=torch.bfloat16)
    vgt = batch.get("video_grid_thw", None)
    if vgt is not None:
        gen_kwargs["video_grid_thw"] = vgt.to(device)
    af = batch.get("audio_feature", None)
    if af is not None:
        gen_kwargs["audio_feature"] = af.to(device=device, dtype=torch.bfloat16)
    al = batch.get("audio_lengths", None)
    if al is not None:
        gen_kwargs["audio_lengths"] = al
    spg = batch.get("second_per_grid_ts", None)
    if spg is not None:
        gen_kwargs["second_per_grid_ts"] = spg.to(device) if torch.is_tensor(spg) else spg

    # [VERIFY] video-SALMONN 2+ generate signature; GDPO uses raw_model.generate(**gen_kwargs).
    out = GEN_MODEL.generate(**gen_kwargs)
    gen_ids = out[0, input_ids.size(1):]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
    return raw, None, None  # post-processing is done by the caller


def render_timeline(segments, max_time):
    fig, ax = plt.subplots(figsize=(8, 1.6))
    ax.set_xlim(0, max(max_time, 0.1))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.hlines(0.5, 0, max_time, color="#cccccc", lw=8, zorder=1)
    colors = plt.cm.tab10.colors
    for i, (s, e) in enumerate(segments):
        ax.hlines(0.5, s, e, color=colors[i % len(colors)], lw=14, zorder=2)
        ax.text((s + e) / 2, 0.78, f"{s:.1f}–{e:.1f}s", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{len(segments)} predicted segment(s)")
    fig.tight_layout()
    return fig


def run(video_path, youtube_url, audio_path, question, max_time, max_new_tokens,
        do_sample, temperature):
    raw, _, _ = infer(video_path, audio_path, question, max_new_tokens, do_sample, temperature,
                      youtube_url=youtube_url)

    segments = parse_multi_segments(raw, max_time=float(max_time))
    answer = clean_text(raw)

    seg_str = "\n".join(f"  • {s:.1f}s – {e:.1f}s" for s, e in segments) or "  (no valid segments)"
    fig = render_timeline(segments, float(max_time))
    return answer, seg_str, fig


# ============================================================
# Gradio UI
# ============================================================
_DEMO_CSS = """
#vid_in {max-height: 240px;}
#vid_in video, #vid_in .video-container {max-height: 220px;}
#aud_in {max-height: 110px;}
#aud_in audio {max-height: 54px;}
"""

with gr.Blocks(title="TiTok — AV-VTG with GDPO", css=_DEMO_CSS) as demo:
    gr.Markdown("## TiTok: Audio-Visual LLM for Multi-Segment Temporal Grounding\n"
                "Upload a video (or a YouTube URL) and ask a question; predicted time segments are shown on a timeline.")
    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Video (upload)", sources=["upload"],
                             height=220, elem_id="vid_in")
            youtube_url = gr.Textbox(
                label="or YouTube URL (used when no video is uploaded)",
                placeholder="https://www.youtube.com/watch?v=...")
            audio = gr.Audio(label="Audio (optional; extracted from the video if empty)",
                             sources=["upload"], type="filepath", elem_id="aud_in")
            question = gr.Textbox(
                label="Question (training phrasing recommended: 'At what point in the video does "
                      "<EVENT> occur in terms of both video and audio?')",
                lines=2,
                placeholder="At what point in the video does a dog barking occur "
                            "in terms of both video and audio?\n"
                            "(the Answer-format instruction is added automatically)")
            max_time = gr.Slider(5, 600, value=60, step=1, label="Video duration (timeline axis, s)")
            with gr.Accordion("Generation settings", open=False):
                max_new = gr.Slider(64, 1024, value=512, step=64, label="max_new_tokens")
                do_sample = gr.Checkbox(value=False, label="sampling (off=greedy)")
                temperature = gr.Slider(0.1, 1.5, value=1.0, step=0.1, label="temperature (when sampling)")
            btn = gr.Button("Run", variant="primary")
        with gr.Column(scale=1):
            answer = gr.Textbox(label="Answer", lines=4)
            seg_box = gr.Textbox(label="Predicted segments", lines=4)
            timeline = gr.Plot(label="Timeline")

    btn.click(run,
              inputs=[video, youtube_url, audio, question, max_time, max_new, do_sample,
                      temperature],
              outputs=[answer, seg_box, timeline])


if __name__ == "__main__":
    demo.queue().launch(share=SHARE, server_name=SERVER_NAME, server_port=SERVER_PORT)
