# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl.txt'.

# ============================================================================
# 3D RoPE (MRoPE) position ids for multimodal inputs.
# Text tokens use a 1D position shared by all three axes; vision/audio tokens use
# (temporal, height, width) positions. Audio blocks and TTI time markers are added here.
#
# - get_rope_index_25 : Qwen2.5-VL with audio and time-token interleaving (TTI)
# - get_rope_index_2  : Qwen2-VL (image/video only)
#
# Both return position_ids (3, batch, seq_len) and mrope_position_deltas
# (per sample: max MRoPE index + 1 - sequence length).
# ============================================================================

import os
import copy
import json
import random
import logging
import re
import time
import math
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import transformers


# [TTI-DBG] throttle counter for the rope entry check (env TTI_DEBUG=1, first N calls)
_TTI_ROPE_DBG_COUNT = 0
_TTI_ROPE_WARN_COUNT = 0


def get_rope_index_25(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    audio_lengths: Optional[list] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    time_token_id_range: Optional[Tuple[int, int]] = None,
    time_marker_token_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.

        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.

        Examples:
            Temporal (Time): 
                3 patches, representing different segments of the video in time.
            Height: 
                2 patches, dividing each frame vertically.
            Width: 
                2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): 
                The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: 
                This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: 
                The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: 
                The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            
            input_ids: 
                [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision position_ids:
                vision temporal position_ids: 
                    [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: 
                    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: 
                    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text position_ids:
                text temporal position_ids: 
                    [101, 102, 103, 104, 105]
                text height position_ids: 
                    [101, 102, 103 104, 105]
                text width position_ids: 
                    [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """

    # Special token ids (fixed by the tokenizer)
    image_token_id = 151655
    video_token_id = 151656
    audio_token_id = 151665
    vision_start_token_id = 151652   # starts a vision/audio block

    mrope_position_deltas = []   # per sample: max MRoPE index + 1 - sequence length

    # Whether TTI (time-marker interleaving) is active. Decided from the marker arguments only,
    # independent of audio (used by the CASE 1 gate below), so video-only + TTI also takes the marker path.
    # - special_token: time_token_id_range=(lo,hi), time_marker_token_len=5
    # - natural_text : time_token_id_range=None,    time_marker_token_len=8
    # - TTI off      : both None
    # Active if either is set. A range without a length (legacy callers) defaults to
    # length 5 for backward compatibility.
    tti_active = (
        (time_marker_token_len is not None and time_marker_token_len > 0)
        or (time_token_id_range is not None)
    )
    if tti_active and (time_marker_token_len is None or time_marker_token_len <= 0):
        time_marker_token_len = 5

    # ==========================================================================
    # CASE 1) audio + video, or video only (+TTI): marker-aware interleaving path
    #   - audio_lengths  : token count of each audio block (0 per block for video-only)
    #   - video_grid_thw : (T, H, W) grid of each video
    #   Video-only inputs also take this path when TTI is active so that markers are placed correctly.
    # ------------------------------------------------------------------
    # Guard: if audio_lengths is None but the input contains audio tokens (the caller did not
    # pass audio_lengths), audio_len=0 would cause a shape mismatch. Such inputs go to CASE 2
    # for backward compatibility; only video-only inputs (no audio tokens) enter this path
    # through tti_active.
    # ==========================================================================
    _no_audio_tokens = (
        input_ids is not None
        and audio_lengths is None
        and not bool((input_ids == audio_token_id).any())
    )
    # Guard: if TTI markers are present but this branch is not taken, marker positions are
    # ignored and temporal grounding silently breaks. Most likely cause: the caller dropped
    # audio_lengths (e.g. a filter that keeps only tensors removes the audio_lengths list).
    _tti_branch = (
        input_ids is not None and video_grid_thw is not None
        and (audio_lengths is not None or (tti_active and _no_audio_tokens))
    )
    # With audio tokens but no audio_lengths, CASE 1 is skipped regardless of TTI and the
    # input falls to CASE 2 (no audio), placing audio tokens like text. This fails silently,
    # so always warn.
    _has_audio_tok = (
        input_ids is not None and audio_lengths is None
        and bool((input_ids == audio_token_id).any())
    )
    if (tti_active or _has_audio_tok) and not _tti_branch:
        global _TTI_ROPE_WARN_COUNT
        if _TTI_ROPE_WARN_COUNT < 5:
            _TTI_ROPE_WARN_COUNT += 1
            _why = ("audio tokens present but audio_lengths not passed"
                    if _has_audio_tok else "markers inserted but the branch was not taken")
            print(f"[TTI-DBG] WARNING: rope CASE 1 not taken: {_why} "
                  f"(tti_active={tti_active}, "
                  f"audio_lengths={'None' if audio_lengths is None else 'ok'}, "
                  f"video_grid_thw={'None' if video_grid_thw is None else 'ok'}) "
                  "-> audio/marker positions are ignored. Check that the caller passes audio_lengths.",
                  flush=True)
    if _tti_branch:
        # video-only + TTI: no audio tokens, so audio_len=0 per video block.
        if audio_lengths is None:
            audio_lengths = [0] * len(video_grid_thw)

        # [TTI-DBG] check that the tti_active branch is entered and markers are detected (first N calls, rank 0).
        #   Can be cross-checked against #markers_in_input from the dataset check.
        global _TTI_ROPE_DBG_COUNT
        if (os.environ.get("TTI_DEBUG", "0").lower() in ("1", "true", "yes", "on")
                and os.environ.get("LOCAL_RANK", "0") == "0"
                and _TTI_ROPE_DBG_COUNT < int(os.environ.get("TTI_DEBUG_SAMPLES", "3"))):
            _TTI_ROPE_DBG_COUNT += 1
            _nmark = 0
            if time_token_id_range is not None:
                _lo, _hi = time_token_id_range
                _flat = input_ids if input_ids.dim() == 1 else input_ids.reshape(-1)
                _nmark = int(((_flat >= _lo) & (_flat <= _hi)).sum().item())
            print(f"[TTI-DBG] ④ rope get_rope_index_25: tti_active={tti_active} "
                  f"time_token_id_range={time_token_id_range} "
                  f"time_marker_token_len={time_marker_token_len} "
                  f"#time_tokens_in_input={_nmark}", flush=True)

        # Interleaved audio/video input
        total_input_ids = input_ids
        # Without attention_mask, treat every token as valid
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        # Output tensor (3, batch, seq_len), filled below
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        video_index = 0   # number of videos processed so far
        attention_mask = attention_mask.to(total_input_ids.device)

        # Iterate over the batch
        for i, input_ids in enumerate(total_input_ids):
            # Drop padding: keep tokens with attention_mask == 1
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0

            # Find every vision_start token; the next token tells image / video / audio apart.
            # TTI special_token: the token after vision_start is <t*> instead of video_pad
            # TTI natural_text : the token after vision_start is plain text ('second' ...)
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            if tti_active:
                if time_token_id_range is not None:
                    # special_token: vision_tokens is video_pad or inside time_token_id_range
                    _lo, _hi = time_token_id_range
                    video_nums = (
                        (vision_tokens == video_token_id)
                        | ((vision_tokens >= _lo) & (vision_tokens <= _hi))
                    ).sum()
                else:
                    # natural_text: a vision_start that is neither image nor audio starts a video block
                    video_nums = (
                        (vision_tokens != image_token_id)
                        & (vision_tokens != audio_token_id)
                    ).sum()
            else:
                video_nums = (vision_tokens == video_token_id).sum()  # videos in this sample

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []   # position-id blocks per segment
            st = 0                         # current start index
            remain_videos = video_nums     # videos left

            # Process one video block at a time
            for _ in range(video_nums):
                # Start of the next video block (text before it)
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                # (T, H, W) grid of this video
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                # Seconds covered by each temporal grid (depends on fps)
                if second_per_grid_ts is not None:
                    second_per_grid_t = second_per_grid_ts[video_index]
                else:
                    second_per_grid_t = 1.0

                # With time tokens, the video block starts at the first marker token of the
                # first chunk, not at the first video_pad.
                ed = ed_video
                if tti_active and ed_video <= len(input_tokens):
                    if time_token_id_range is not None:
                        # special_token: first time token in the prefix
                        lo, hi = time_token_id_range
                        prefix = input_ids[st:ed_video]
                        time_mask_prefix = (prefix >= lo) & (prefix <= hi)
                        first_time = torch.nonzero(time_mask_prefix, as_tuple=False)
                        if first_time.numel() > 0:
                            ed = st + first_time[0].item()
                    else:
                        # natural_text: markers have a fixed length, so N tokens before the first video_pad
                        ed = ed_video - time_marker_token_len

                # Grid seen by the LLM: H and W are reduced by spatial_merge_size.
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st  # text length before the video block

                # Positions continue from the previous block's max + 1
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                # Text part: all three axes share a 1D counter
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                # ------- 3D positions of the video tokens -------
                # t axis: frame index (0..T-1) * (second_per_grid * 2),
                #   i.e. the elapsed time is encoded in the position
                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                time_tensor = expanded_range * second_per_grid_t * 2
                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                # h axis: 0..H-1 expanded to (T,H,W) and flattened
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                # w axis: 0..W-1 expanded to (T,H,W) and flattened
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                # (3, T*H*W) positions for all video tokens
                video_pos = torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                # llm_pos_ids_list.append(video_pos)  # appended together with audio below

                # ------- positions of the audio tokens -------
                audio_len = audio_lengths[video_index]  # audio tokens in this block

                # Build audio positions on the CPU like the rest of this function;
                # everything is moved to position_ids.device at the end.
                time_index_audio = torch.arange(audio_len)
                w_index_audio = time_index_audio                      # w grows with time
                h_index_audio = torch.zeros_like(time_index_audio)    # h fixed at 0 (1D signal)
                audio_pos = torch.stack([time_index_audio, h_index_audio, w_index_audio]) + st_idx + text_len

                # ------- reorder positions for the interleaved video/audio block -------
                # Fill positions where audio / video / time-marker tokens appear in input_ids.
                if tti_active:
                    # All marker tokens of chunk k share one 3D position:
                    #   t = w = k * second_per_grid_t * 2  (same as the first frame of the chunk)
                    #   h = 0
                    # Token ids (special_token) or text content (natural_text) already order the
                    # tokens inside a marker, so RoPE does not separate them.
                    M = time_marker_token_len  # special_token=5, natural_text=9
                    chunk_t_base = (
                        torch.arange(llm_grid_t) * second_per_grid_t * 2
                    ).long()                                           # (T,)
                    time_t_flat = (
                        chunk_t_base.unsqueeze(1).expand(-1, M).flatten()
                    )                                                  # (M*T,)
                    time_w_flat = time_t_flat.clone()
                    time_h_flat = torch.zeros_like(time_t_flat)
                    time_pos = (
                        torch.stack([time_t_flat, time_h_flat, time_w_flat])
                        + st_idx + text_len
                    )

                    # Block size = markers M*T + video T*H*W/merge^2 + audio audio_len
                    st = ed + M * llm_grid_t + llm_grid_t * llm_grid_h * llm_grid_w + audio_len
                    # Compare / index on the CPU; everything moves to the GPU at the end.
                    block_ids = input_ids[ed:st].cpu()
                    audio_visual_pos = torch.zeros_like(
                        torch.cat((video_pos, audio_pos, time_pos), dim=1)
                    )
                    is_video = block_ids == video_token_id
                    is_audio = block_ids == audio_token_id
                    if time_token_id_range is not None:
                        # special_token: markers identified by id range
                        lo, hi = time_token_id_range
                        is_time = (block_ids >= lo) & (block_ids <= hi)
                    else:
                        # natural_text: tokens in the block that are neither video nor audio are markers
                        is_time = ~(is_video | is_audio)
                    audio_visual_pos[:, is_audio] = audio_pos
                    audio_visual_pos[:, is_video] = video_pos
                    audio_visual_pos[:, is_time] = time_pos
                    llm_pos_ids_list.append(audio_visual_pos)
                else:
                    audio_visual_pos = torch.zeros_like(torch.cat((video_pos, audio_pos), dim=1))
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w + audio_len
                    # Compare input_ids[ed:st] on the CPU as well.
                    input_ids_cpu_slice = input_ids[ed:st].cpu()
                    audio_visual_pos[:, input_ids_cpu_slice == audio_token_id] = audio_pos
                    audio_visual_pos[:, input_ids_cpu_slice == video_token_id] = video_pos
                    llm_pos_ids_list.append(audio_visual_pos)

                video_index += 1
                remain_videos -= 1

            # Remaining text after the last video gets 1D positions
            if st < len(input_tokens):
                if len(llm_pos_ids_list) > 0:
                    # continue from the max of the last block's time axis + 1
                    last_time_max = llm_pos_ids_list[-1][0].max()
                    st_idx = last_time_max + 1
                else:
                    st_idx = 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            # Concatenate the position blocks along seq_len
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            # Write only at non-padding positions
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            # max MRoPE index + 1 - original sequence length for this sample
            # (used to correct cache_position during generation)
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas

    # ==========================================================================
    # CASE 2) image or video only (no audio): original Qwen2.5-VL path
    # ==========================================================================
    elif input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        # Iterate over the batch
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]   # drop padding
            image_nums, video_nums = 0, 0

            # Count images/videos from the token after each vision_start
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            # Process image/video blocks in order of appearance
            for _ in range(image_nums + video_nums):
                # Find the next image and video token; handle whichever comes first
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    # image block
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0   # images have no time axis
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    # video block
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                # LLM grid after merging
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st   # text length before the vision block

                # next start position (previous block max + 1)
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                # text positions (same on all three axes)
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                # 3D positions of the vision (image/video) block
                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                # time axis: tokens per second * temporal patch (=2)
                time_tensor = expanded_range * second_per_grid_t * 2
                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                # height / width axes flattened in T,H,W order
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                # append the (3, T*H*W) vision positions
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                # next start = after the vision tokens
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # trailing text after the last vision block
            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            # concatenate into the final position_ids
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas

    # ==========================================================================
    # CASE 3) audio only (no image/video)
    # ==========================================================================
    elif input_ids is not None and audio_lengths is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        audio_index = 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]   # drop padding
            audio_nums = 0
            # Find audio starts (audio reuses the vision_start token)
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            audio_nums = (vision_tokens == audio_token_id).sum()   # audio blocks in this sample
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_audios = audio_nums

            # Process one audio block at a time
            for _ in range(audio_nums):
                # Start of the current audio token block
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio = input_tokens.index(audio_token_id, st)
                else:
                    ed_audio = len(input_tokens) + 1
                ed = ed_audio

                # audio_lengths are given in tokens
                llm_grid_t = audio_lengths[audio_index]  # tokens along time
                llm_grid_h = 1                           # audio has no height
                llm_grid_w = 1                           # audio has no width

                text_len = ed - st
                st_idx = (llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0)
                # text positions
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                # 3-axis positions of the audio block
                # (t: 0..L-1, h: same as t, w: 0) for L tokens
                time_index = torch.arange(llm_grid_t, device=input_ids.device)
                h_index = time_index
                w_index = torch.zeros_like(time_index)
                audio_pos = torch.stack([time_index, h_index, w_index]) + st_idx + text_len
                llm_pos_ids_list.append(audio_pos)

                st = ed + llm_grid_t    # move past the block
                audio_index += 1
                remain_audios -= 1

            # trailing text
            if st < len(input_tokens):
                if len(llm_pos_ids_list) > 0:
                    # continue from the max of the last audio block's time axis
                    last_time_max = llm_pos_ids_list[-1][0].max()
                    st_idx = last_time_max + 1
                else:
                    st_idx = 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            # concatenate all segments
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas

    # ==========================================================================
    # CASE 4) no multimodal input (text only):
    #  plain 1D positions copied to all three axes.
    # ==========================================================================
    else:
        if attention_mask is not None:
            # positions from the cumulative sum of attention_mask - 1
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)   # padding positions set to 1
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            # without attention_mask: 0..seq-1
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


# ============================================================================
# Qwen2-VL 3D RoPE index (kept for the Qwen2-VL model path).
# Same structure as get_rope_index_25 without audio, and the temporal position
# is the plain frame index 0,1,2,... instead of a multiple of second_per_grid_t * 2.
# ============================================================================
def get_rope_index_2(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    time_token_id_range: Optional[Tuple[int, int]] = None,  # API compatibility (no time tokens on the Qwen2-VL path)
    time_marker_token_len: Optional[int] = None,            # API compatibility (no time tokens on the Qwen2-VL path)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embeddin for text part.
        Examples:
            Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [3, 4, 5, 6, 7]
            text height position_ids: [3, 4, 5, 6, 7]
            text width position_ids: [3, 4, 5, 6, 7]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    # Special token ids (Qwen2-VL)
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []

    # Multimodal input with images or videos
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        # position_ids: (3, batch, seq_len)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]   # drop padding
            image_nums, video_nums = 0, 0

            # Count images/videos from the token after each vision_start
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            # Process image/video blocks in order of appearance
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                # handle whichever block comes first
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                # grid seen by the LLM (H/W after merging)
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                # continue from the previous block max + 1
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                # text positions (same on all three axes)
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                # vision positions: Qwen2-VL uses 0..T-1 on the time axis
                t_index = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # trailing text
            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            # concatenate into the final position ids
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        # text-only input: 1D positions expanded to (3, ...)
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas
