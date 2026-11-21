#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preflight_tti_off.py — w/o TTI(tti_mode=off) 학습 시작 전 검증. **모델 로드 없음** (CPU 로 충분).

  gdpo_trainer_batch.py 의 tti_mode=off 경로가 실제로 "마커 없음 + rope OFF" 인지 확인한다.
    (a) 모델 config.json 의 time_token_id_range / time_marker_token_len 이 None
        (트레이너는 off 면 강제로 None 으로 덮지만, 원본 상태도 기록)
    (b) 같은 샘플을 off / special_token(on) 두 판으로 만들어 비교:
        - vision span(<|vision_start|>..<|vision_end|>) 안의 time 토큰 수 → off 는 0, on 은 > 0
        - video_pad / audio_pad 개수                              → 두 판 동일 (마커만 빠져야 함)
        - 입력 길이 차이                                           → on - off == #markers(on)
        - off 의 position_ids 가 rope 에 time 인자 None 으로 계산됨 (dataset.time_token_id_range=None)
    오디오 샘플(unav)과 video-only 샘플(charades)을 모두 확인한다 (dataset.py 두 분기).

  사용: python gdpo/preflight_tti_off.py [--n 3]
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "video_SALMONN2_plus"))

from qwenvl.data.dataset import LazySupervisedDataset  # noqa: E402
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast  # noqa: E402
from transformers import AutoTokenizer, WhisperFeatureExtractor  # noqa: E402

SFT_CKPT = os.environ.get("SFT_CKPT", "/workspace/checkpoints/base/salmonn2p_7b_unav_v8")
DATASET = os.environ.get("DATASET", "/workspace/data/uncha_v3_local.json")

VIDEO_PAD, AUDIO_PAD, VISION_START, VISION_END = 151656, 151665, 151652, 151653


@dataclass
class DA:  # gdpo_trainer_batch.py GDPODataArgs 와 동일 값
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
    tti_time_format: str = "off"
    feature_size: int = 128
    chunk_length: int = 30
    hop_length: int = 160
    sampling_rate: int = 16000
    image_processor: object = None
    audio_processor: object = None
    video_cache_size: int = 0


def build(tok, ip, ap, path, fmt):
    da = DA(dataset_use=path, tti_time_format=fmt, image_processor=ip, audio_processor=ap)
    return LazySupervisedDataset(tokenizer=tok, data_args=da)


def stats(item, id_range):
    ids = item["input_ids"][0].tolist()
    lo, hi = id_range
    s, e = ids.index(VISION_START), ids.index(VISION_END)
    span = ids[s:e + 1]
    return {
        "len": len(ids),
        "markers_in_vision": sum(lo <= t <= hi for t in span),
        "video_pad": ids.count(VIDEO_PAD),
        "audio_pad": ids.count(AUDIO_PAD),
    }


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--n", type=int, default=2, help="분기(오디오/비디오전용)별 검사 샘플 수")
    args = ap_.parse_args()

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok &= bool(cond)

    print(f"[preflight] SFT_CKPT={SFT_CKPT}\n[preflight] DATASET={DATASET}")
    cfg = json.load(open(os.path.join(SFT_CKPT, "config.json")))
    print(f"(a) config.json: time_token_id_range={cfg.get('time_token_id_range')} "
          f"time_marker_token_len={cfg.get('time_marker_token_len')}")

    tok = AutoTokenizer.from_pretrained(SFT_CKPT, model_max_length=10000, padding_side="left")
    t0, tdot = tok.convert_tokens_to_ids("<t0>"), tok.convert_tokens_to_ids("<tdot>")
    id_range = (min(t0, tdot), max(t0, tdot))
    ip = Qwen2VLImageProcessorFast.from_pretrained(SFT_CKPT)
    fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=30)

    ds_off = build(tok, ip, fe, DATASET, "off")
    ds_on = build(tok, ip, fe, DATASET, "special_token")
    check(ds_off.time_token_id_range is None and not ds_off.time_marker_token_len,
          f"dataset(off): time_token_id_range={ds_off.time_token_id_range}, "
          f"time_marker_token_len={ds_off.time_marker_token_len} → rope OFF 인자")

    raw = ds_off.list_data_dict
    picks = {"audio+video": [i for i, x in enumerate(raw) if x.get("audio")][: args.n],
             "video-only": [i for i, x in enumerate(raw) if not x.get("audio")][: args.n]}

    print(f"(b) off vs on 비교 (time-token id_range={id_range})")
    for branch, idxs in picks.items():
        for i in idxs:
            so, sn = stats(ds_off[i], id_range), stats(ds_on[i], id_range)
            print(f" [{branch}] idx={i} {os.path.basename(raw[i]['video'])}  off={so}  on={sn}")
            check(so["markers_in_vision"] == 0, "off: vision span 내 time marker 0개")
            check(sn["markers_in_vision"] > 0, "on : vision span 내 time marker 존재 (대조군)")
            check(so["video_pad"] == sn["video_pad"] and so["audio_pad"] == sn["audio_pad"],
                  "video_pad/audio_pad 개수 off==on")
            check(sn["len"] - so["len"] == sn["markers_in_vision"],
                  f"길이차 {sn['len'] - so['len']} == on 마커수 (마커 외 차이 없음)")
            if branch == "video-only":
                check(so["audio_pad"] == 0, "video-only: audio_pad 0")

    print("\n[preflight] " + ("✅ ALL PASS — tti_mode=off 정상" if ok else "❌ FAIL — 위 항목 확인"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
