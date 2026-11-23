#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TiTok stage 2: GDPO trainer for video-SALMONN 2+ (Qwen2.5-VL backbone).

Optimizes the multi-segment temporal grounding policy with group reward-decoupled
policy optimization (GDPO): each reward channel is normalized within its rollout
group before the channels are combined into one advantage.

Main features (all controlled by the YAML config or CLI flags):
  - Reward channels are defined by --reward_module. The paper uses
    reward_functions_rM_sep3 with five channels: format, count, global, local,
    precision.
  - Clip-higher clipped surrogate (DAPO): -min(r*A, clip(r, 1-eps_low, 1+eps_high)*A).
    It only binds when clip.num_iterations (mu) > 1, i.e. when one rollout is
    reused for mu policy updates.
  - A block-repeat sampler keeps optimizer steps between the mu iterations, so
    gradient accumulation can be used with clip-higher.
    Effective batch = gradient_accumulation_steps x num_gpus.
  - Rollouts are cached by prompt signature for reuse across the mu iterations.
  - gdpo.multi_seg_weight scales the advantage of prompts with >= 2 GT segments
    (1.0 = off; the paper uses 1.0).
  - Optional validation on held-out sets (config.val) logged to val_metrics.jsonl.

Training paths (chosen automatically from --model_path):
  (A) continue an SFT LoRA adapter: model_path = adapter, model_base = time-token
      base; set lora.enabled=false.
  (B) fresh LoRA on a merged SFT model: model_path = model_base = merged model;
      set lora.enabled=true. The reference policy is the merged SFT model.
      The paper uses path (B).

Usage (see gdpo/run_rMsep3_uncha_noscaling_1gpu.sh for the paper run):
  torchrun --standalone --nproc_per_node=1 gdpo/gdpo_trainer_batch.py       --config        gdpo/config_sep3_uncha_1gpu.yaml       --model_path    $CKPT_DIR/base/salmonn2p_7b_unav_v8       --model_base    $CKPT_DIR/base/salmonn2p_7b_unav_v8       --dataset_path  $DATA_DIR/gdpo/train.json       --reward_module reward_functions_rM_sep3 --tti_mode on       --output_dir    $CKPT_DIR/gdpo/titok_gdpo

CLI arguments (CLI > YAML config > built-in defaults):
  --config                  YAML config; ${VAR} values are expanded from the environment.
  --model_path              [required] SFT adapter (path A) or merged SFT model (path B).
  --model_base              time-token base model.
  --dataset_path            [required] training JSON.
  --output_dir              checkpoint / log directory.
  --max_steps   int         > 0 overrides num_train_epochs.
  --run_name    str         W&B run name (default: output_dir basename).
  --tti_mode    off|on      time-token interleaving; must match the data and model.
  --reward_module str       reward module name. A module may declare
                            REWARD_CHANNELS = [(name, fn, needs_gt), ...].
  --val_path                force validation on with this JSON.
  --resume_from_checkpoint  'True' = latest checkpoint in output_dir, or a path.

Debug environment variables:
  TTI_DEBUG=1 [TTI_DEBUG_STEPS=3]   print TTI consistency checks on rank 0 for the first N steps.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Callable, List, Dict, Optional, Union

import yaml
import torch
from packaging import version

import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
    PreTrainedModel,
    Trainer,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig





# project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# video-SALMONN 2+ model
sys.path.insert(0, os.path.join(PROJECT_ROOT, "video_SALMONN2_plus"))
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from qwenvl.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset



# Reward functions: the module is chosen at runtime with --reward_module (load_reward_module),
# so the trainer does not have to be duplicated per reward.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)


def load_reward_module(module_name: str):
    """Dynamically load a reward module and return its channel spec [(name, fn, needs_gt), ...].

    Two supported module conventions:
        (1) REWARD_CHANNELS (recommended; any number of channels)
                    If the module defines REWARD_CHANNELS = [(name:str, fn:callable, needs_gt:bool), ...],
                    it is used as is; the module fully decides channel count, names and order.
                    e.g. reward_functions_rM_sep3 -> [format, count, global, local, precision]
                    (needs_gt=True calls fn(completion, gt), False calls fn(completion).)
        (2) naming convention (legacy)
                    Without REWARD_CHANNELS, channels are assembled from function names:
                        required: format_reward, iou_reward
                        optional: timestamp_reward, modality_reward (ignored if missing)
                    e.g. reward_functions

    The trainer works with any returned spec:
        - reward_weights are resized to the number of channels in main()
        - W&B metrics are logged per channel __name__ (rewards/<name>)
    so adding a reward module with new channels needs no trainer change.
    """
    import importlib
    mod = importlib.import_module(module_name)

    # (1) the module declares its channel spec: use it as is
    raw = getattr(mod, "REWARD_CHANNELS", None)
    if raw is not None:
        specs = []
        for ch in raw:
            if not (isinstance(ch, (tuple, list)) and len(ch) == 3):
                raise ValueError(
                    f"[GDPO] {module_name}.REWARD_CHANNELS entries must be (name, fn, needs_gt) "
                    f"3-tuples: {ch!r}")
            name, fn, needs_gt = ch
            if not callable(fn):
                raise ValueError(f"[GDPO] fn of reward channel '{name}' is not callable: {fn!r}")
            specs.append((str(name), fn, bool(needs_gt)))
        if not specs:
            raise ValueError(f"[GDPO] {module_name}.REWARD_CHANNELS is empty")
        return specs

    # (2) legacy naming-convention fallback
    specs = [("format", mod.format_reward, False),   # (name, fn, needs_gt)
             ("iou", mod.iou_reward, True)]
    for name in ("timestamp", "modality"):  # optional (CoT)
        fn = getattr(mod, f"{name}_reward", None)
        if fn is not None:
            specs.append((name, fn, True))
    return specs

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


# ============================================================
# TTI debug instrumentation (env TTI_DEBUG=1; rank 0 only, first N steps).
#   No effect on normal runs (off).
# ============================================================
TTI_DEBUG = os.environ.get("TTI_DEBUG", "0").lower() in ("1", "true", "yes", "on")
TTI_DEBUG_STEPS = int(os.environ.get("TTI_DEBUG_STEPS", "3"))  # only the first N compute_loss steps


def _is_rank0() -> bool:
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"


def _tti_dbg(msg: str):
    """Print only when TTI_DEBUG is on and on rank 0."""
    if TTI_DEBUG and _is_rank0():
        print(f"[TTI-DBG] {msg}", flush=True)


# Dataset tag for per-dataset reward logging: W&B keys rewards/<channel>/<tag>.
DATASET_TAGS = ["unav", "charades", "puvalor"]


def _dataset_tag(video_path) -> str:
    """Dataset tag from the video path (/datasets/<name>/ or a <name>/ prefix):
        unav_100 -> unav / charades_sta -> charades / pu-valor_2900 -> puvalor / otherwise unknown."""
    if not video_path:
        return "unknown"
    p = str(video_path)
    name = p.split("/datasets/", 1)[1].split("/", 1)[0] if "/datasets/" in p else ""
    if name.startswith("unav"):
        return "unav"
    if name.startswith("charades"):
        return "charades"
    if name.startswith("pu-valor") or name.startswith("puvalor"):
        return "puvalor"
    return "unknown"


def count_time_markers(ids, id_range) -> int:
    """Number of tokens in ids (1D tensor or list) inside the time-token id range [lo, hi]."""
    if id_range is None:
        return 0
    lo, hi = id_range
    if isinstance(ids, torch.Tensor):
        return int(((ids >= lo) & (ids <= hi)).sum().item())
    return sum(1 for x in ids if lo <= int(x) <= hi)


# -- resume: RNG state loading workaround (torch>=2.4 defaults to weights_only=True) --
# rng_state_*.pth stores np.random.get_state() (numpy objects), which
# torch.load(weights_only=True) refuses to unpickle, so transformers._load_rng_state
# crashes with UnpicklingError on resume. These are our own trusted checkpoints,
# so the numpy globals are allow-listed.
try:
    import numpy as _np
    import torch.serialization as _ts
    import _codecs as _cod
    # _codecs.encode is part of the serialized numpy random state
    _safe = [_np.ndarray, _np.dtype, _cod.encode]
    for _m in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            _mod = __import__(_m, fromlist=["_reconstruct", "scalar"])
            _safe += [getattr(_mod, "_reconstruct", None), getattr(_mod, "scalar", None)]
        except Exception:
            pass
    try:
        import numpy.dtypes as _ndt
        _safe += [getattr(_ndt, _n) for _n in dir(_ndt) if _n.endswith("DType")]
    except Exception:
        pass
    _ts.add_safe_globals([x for x in _safe if x is not None])
except Exception as _e:
    print(f"[GDPO] add_safe_globals(numpy) failed (ignored): {_e}")

# -- last resort: numpy dtypes not covered by add_safe_globals (e.g. numpy.dtype[uint32]) --
# For our own trusted checkpoints only: if weights_only unpickling fails, retry once with
# weights_only=False (avoids rng_state_*.pth crashes on resume across torch versions).
_orig_torch_load = torch.load
def _torch_load_trusted(*args, **kwargs):
    try:
        return _orig_torch_load(*args, **kwargs)
    except Exception as _e:
        _m = str(_e)
        if "Weights only load failed" in _m or "WeightsUnpickler" in _m:
            kwargs["weights_only"] = False
            return _orig_torch_load(*args, **kwargs)
        raise
torch.load = _torch_load_trusted





# ============================================================
# Utilities
# ============================================================

def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])

def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])





# ============================================================
# RepeatSampler (rollout reuse for mu > 1)
# ============================================================

class _RepeatSampler:
    """Wrap a base sampler and yield each index num_repeat times in a row.

    With mu > 1 the dataloader yields (i,i,j,j,...), so compute_loss generates a rollout
    only on the first call and reuses the cache for the remaining mu-1 calls without
    skipping distinct samples (same idea as TRL GRPOTrainer's RepeatSampler).
    """
    def __init__(self, base_sampler, num_repeat: int):
        self.base_sampler = base_sampler
        self.num_repeat = max(1, int(num_repeat))

    def __len__(self):
        return len(self.base_sampler) * self.num_repeat

    def __iter__(self):
        for idx in self.base_sampler:
            for _ in range(self.num_repeat):
                yield idx


class _BlockRepeatSampler:
    """Group the base sampler into blocks of block_size (N) and repeat each block num_repeat (mu) times:
          [p0..p_{N-1}] x mu, [pN..p_{2N-1}] x mu, ...  (a trailing partial block < N is dropped)

    With grad_accum=N the microbatch order becomes i,j,i,j (block repeat), so HF calls
    optimizer.step() every N microbatches and the policy is actually updated between the mu
    iterations (ratio r != 1 for the PPO/GRPO clip). N=1 equals _RepeatSampler (each index
    mu times); mu=1 gives one pass in N-chunks = standard grad_accum."""
    def __init__(self, base_sampler, block_size: int, num_repeat: int):
        self.base_sampler = base_sampler
        self.N = max(1, int(block_size))
        self.mu = max(1, int(num_repeat))

    def __len__(self):
        n_full = len(self.base_sampler) // self.N  # drop the partial block
        return n_full * self.N * self.mu

    def __iter__(self):
        block = []
        for idx in self.base_sampler:
            block.append(idx)
            if len(block) == self.N:
                for _ in range(self.mu):
                    for j in block:
                        yield j
                block = []
        # drop the trailing partial block (< N) to keep the cache aligned


# ============================================================
# GDPOTrainer
# ============================================================

class GDPOTrainer(Trainer):
    """GDPO trainer for video-SALMONN 2+ (Qwen2.5-VL backbone)."""

    def __init__(
        self,
        model: PreTrainedModel,
        reward_funcs: list[RewardFunc],
        args: GRPOConfig = None,
        train_dataset=None,
        processing_class=None,
        ref_model: Optional[PreTrainedModel] = None,
        callbacks=None,
        optimizers=(None, None),
        peft_config: Optional["PeftConfig"] = None,
        reward_weights: Optional[list[float]] = None,
    ):
        if args is None:
            model_name = model.config._name_or_path.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GDPO")


        # PEFT
        # The LoRA adapter already targets the LLM,
        # so the encoders do not need to be split off.
        if peft_config is not None:
            model = get_peft_model(model, peft_config)
            # With gradient checkpointing and a frozen base, re-enable requires_grad on the
            # embedding outputs so gradients reach the fresh LoRA (required for path B).
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()



        # Reference model
        self.beta = args.beta
        # For a PEFT model, disable_adapter() acts as the reference model (saves ~14 GB);
        # compute_loss already has the disable_adapter branch.
        _is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        if self.beta == 0.0:
            self.ref_model = None
        elif ref_model is not None:
            self.ref_model = ref_model
        elif _is_peft_model:
            # With PEFT, the disabled adapter is the reference (saves memory)
            print("[GDPO] PEFT model detected → ref_model=None (use disable_adapter path)")
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = video_SALMONN2_plus.from_pretrained(
                model.config._name_or_path,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
            )
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None



        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.config._name_or_path, padding_side="left"
            )
        pad_token_id = processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = processing_class.eos_token_id


        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs

        if reward_weights is not None:
            self.reward_weights = torch.tensor(reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)


        # Generation config
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = getattr(args, "temperature", 1.0)
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=self.temperature,
            top_p=1.0,
            top_k=50,  # override the base config top_k=1 (diversity)
            repetition_penalty=1.0,  # no repetition penalty on time tokens
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )


        # ── DAPO clip-higher (ε_high > ε_low) ──────────────────────────────
        # clipped surrogate:  loss = -min( r·A,  clip(r, 1-ε_low, 1+ε_high)·A )
        #   r = pi_theta(o)/pi_old(o) = exp(logp - old_logp). A larger eps_high (looser upper clip)
        #   leaves room for low-probability tokens to grow, keeping exploration (clip-higher).
        # DAPO recommends eps_low=0.2 / eps_high=0.28; injected into args from config (clip.*).
        self.epsilon_low = getattr(args, "epsilon", 0.2)
        self.epsilon_high = getattr(args, "epsilon_high", None) or 0.28
        # Whether to use the clip-higher path (the default behaviour of this trainer).
        self.clip_higher = getattr(args, "clip_higher", True)
        self.use_grpo = getattr(args, "use_grpo", True)

        # -- mu (num_iterations): reuse one rollout for mu policy updates --
        # Only with mu > 1 do the old (generation-time snapshot) and current policies differ, so
        # r = exp(new - old) != 1 and clip-higher binds. mu=1 regenerates every step (plain GDPO).
        self.num_iterations = max(1, int(getattr(args, "num_iterations", 1)))

        # -- grad_accum support (effective batch >= N x num_gpu) --
        # The block-repeat sampler orders microbatches i,j,i,j; HF steps the optimizer every N
        # (grad_accum) microbatches, so the policy is updated between the mu iterations and
        # r != 1 for the clip. The cache is keyed by a prompt signature (no slots/counters), so
        # DDP reordering cannot cause wrong reuse or crashes (signature mismatch = miss -> regenerate).
        #   _rollout_cache: sig -> {"R": rollout, "uses": int}. Entries reaching mu are evicted at once.
        self.grad_accum = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
        self._rollout_cache = {}

        # -- advantage weight for multi-segment (GT >= 2) prompts --
        # GDPO normalizes advantages within each prompt group (one microbatch = one prompt), so the
        # absolute reward cannot express that multi-segment prompts matter more. Instead the
        # normalized advantage of multi-GT steps is multiplied by w, offsetting the single-segment
        # prior of the data. w=1.0 disables it (the paper uses 1.0).
        # Works with and without clip-higher: both loss paths use R['advantages'] from _generate_rollout.
        self.multi_seg_weight = float(getattr(args, "multi_seg_weight", 1.0))  # 1.0 = off (default)


        # Data collator — VS2+
        # LazySupervisedDataset (replaces video-SALMONN 2's av_dataset.py)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=processing_class)


        # logging
        model.warnings_issued["estimate_tokens"] = True

        # metrics
        self._metrics = defaultdict(list)



        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )



    # ============================================================
    # Trainer override: disabled
    # ============================================================

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _prepare_inputs(self, inputs):
        return inputs

    def _get_train_sampler(self, *args, **kwargs):
        # Feed batches in block-repeat order.
        #   accelerate distributes batches to ranks round-robin (BatchSamplerShard,
        #   split_batches=False). With a global block_size of only grad_accum, each rank would
        #   receive [a,a] (adjacent repeats) after sharding, grad_accum would put a_fresh and
        #   a_cached in the same optimizer step, r would stay ~1 (clip ineffective) and the
        #   effective batch would not grow (duplicate prompts).
        #   With block_size = grad_accum x num_processes, after round-robin sharding each rank
        #   receives exactly [p0..p_{N-1}] x mu (N = grad_accum, distinct prompts):
        #        global [a,b,c,d]×μ  --round-robin(G=2)-->  rank0=[a,c]×μ, rank1=[b,d]×μ
        #   -> policy updates between mu iterations (clip r != 1) and effective batch = grad_accum x num_processes.
        #   (With 1 GPU this is block_size = grad_accum.)
        base = super()._get_train_sampler(*args, **kwargs)
        if base is not None and (self.num_iterations > 1 or self.grad_accum > 1):
            world = max(1, int(getattr(self.accelerator, "num_processes", 1)))
            return _BlockRepeatSampler(base, block_size=self.grad_accum * world,
                                       num_repeat=self.num_iterations)
        return base

    def _prompt_sig(self, inputs):
        """Lightweight prompt signature (collisions practically impossible); used as cache key, order independent."""
        t = inputs["input_ids"][0]
        return (int(t.shape[0]), int(t.sum().item()),
                int(t[::7].sum().item()), int(t.float().square().sum().item()))

    # ============================================================
    # Per-token log probabilities (VS2+ API)
    # ============================================================

    def _get_per_token_logps(
        self, model, input_ids, attention_mask,
        pixel_values_videos=None, video_grid_thw=None,
        audio_feature=None, audio_lengths=None,
        position_ids=None, second_per_grid_ts=None,
        inputs_embeds=None, return_entropy=False,
    ):
        """Calls video-SALMONN 2+ sft_forward.
        → logits
        → per-token log probability.

        If inputs_embeds is given, the video/audio encoders are not re-run (cached embeddings are reused).
        input_ids/video_grid_thw/audio_lengths are still passed for the RoPE index computation.
        """
        # With inputs_embeds, pixel_values_videos/audio_feature are not passed, so sft_forward skips re-encoding
        if inputs_embeds is not None:
            pixel_values_videos = None
            audio_feature = None
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            audio_feature=audio_feature,
            audio_lengths=audio_lengths,
            second_per_grid_ts=second_per_grid_ts,
            inputs_embeds=inputs_embeds,
            train_type="sft",
            use_cache=False,
        ).logits

        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        # align lengths
        min_len = min(logits.size(1), input_ids.size(1))
        logits = logits[:, -min_len:, :]
        input_ids = input_ids[:, -min_len:]
        vocab_size = logits.size(-1)
        input_ids = input_ids.clamp(0, vocab_size - 1)

        per_token_logps = []
        entropies = [] if return_entropy else None
        for logits_row, input_ids_row in zip(logits, input_ids):
            # Saves memory: logp = logit - logsumexp without keeping a full log_softmax([seq, vocab]) copy
            token_logit = torch.gather(
                logits_row, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_logit - torch.logsumexp(logits_row, dim=-1))
            if return_entropy:
                # Policy entropy H = -sum p*logp per position over the vocab. Logging only, detached.
                log_probs = logits_row.log_softmax(dim=-1)
                entropies.append((-(log_probs.exp() * log_probs).sum(dim=-1)).detach())
                del log_probs

        logps = torch.stack(per_token_logps)
        if return_entropy:
            return logps, torch.stack(entropies)
        return logps




    # ============================================================
    # GDPO advantage computation
    # ============================================================
    def _compute_gdpo_advantages(self, rewards_per_func, rewards):
        num_funcs = rewards_per_func.shape[1]
        device = rewards_per_func.device

        if num_funcs <= 1:
            mean = rewards.view(-1, self.num_generations).mean(dim=1)
            std = rewards.view(-1, self.num_generations).std(dim=1)
            mean = mean.repeat_interleave(self.num_generations, dim=0)
            std = std.repeat_interleave(self.num_generations, dim=0)
            return (rewards - mean) / (std + 1e-4)

        reward_weights = self.reward_weights.to(device)
        rewards_per_func = torch.nan_to_num(rewards_per_func)

        all_adv = []
        for i in range(num_funcs):
            r_i = rewards_per_func[:, i]
            grouped = r_i.view(-1, self.num_generations)
            mean_g = grouped.mean(dim=1, keepdim=True)
            std_g = grouped.std(dim=1, keepdim=True)
            normed = ((grouped - mean_g) / (std_g + 1e-4)).view(-1)
            all_adv.append(normed)

        stacked = torch.stack(all_adv, dim=1)
        combined = (stacked * reward_weights.unsqueeze(0)).sum(dim=1)
        advantages = (combined - combined.mean()) / (combined.std() + 1e-4)
        return advantages



    # ============================================================
    # compute_loss 
    # ============================================================

    def _completion_logps(self, model, R, raw_encode_model, return_entropy=False):
        """Per-token logps for the cached completions.
        Whether gradients flow (no_grad / grad) is decided by the caller's context.
        With return_entropy=True returns (logps, entropy[num_gen, comp_len]) (logging only, detached)."""
        _VIDEO_TOKEN_ID = 151656
        comp_len = R["comp_len"]
        all_logps = []
        all_ent = [] if return_entropy else None
        for g in range(self.num_generations):
            pc_ids = torch.cat([R["prompt_ids_single"], R["completion_ids_clean"][g:g + 1]], dim=1)
            pc_mask = torch.cat([R["prompt_mask_single"], R["completion_mask"][g:g + 1]], dim=1)
            # Text positions use the trainable embed_tokens (grad); video/audio positions use the cached frozen embeddings.
            text_embeds = raw_encode_model.model.embed_tokens(pc_ids)
            if R["cached_video_embeds"] is not None:
                mask = (pc_ids == _VIDEO_TOKEN_ID)
                mask_exp = mask.unsqueeze(-1).expand_as(text_embeds)
                v = R["cached_video_embeds"].to(text_embeds.device, text_embeds.dtype)
                text_embeds = text_embeds.masked_scatter(mask_exp, v)
            if R["cached_audio_embeds"] is not None and R["audio_token_id_cfg"] is not None:
                mask = (pc_ids == R["audio_token_id_cfg"])
                mask_exp = mask.unsqueeze(-1).expand_as(text_embeds)
                a = R["cached_audio_embeds"].to(text_embeds.device, text_embeds.dtype)
                text_embeds = text_embeds.masked_scatter(mask_exp, a)
            _out = self._get_per_token_logps(
                model, pc_ids, pc_mask,
                video_grid_thw=R["video_grid_thw"],
                audio_lengths=R["audio_lengths"],
                second_per_grid_ts=R["second_per_grid_ts"],
                inputs_embeds=text_embeds,
                return_entropy=return_entropy,
            )
            if return_entropy:
                g_logps, g_ent = _out
                g_ent = g_ent[:, -comp_len:] if comp_len > 0 else g_ent[:, :0]
                all_ent.append(g_ent)
            else:
                g_logps = _out
            g_logps = g_logps[:, -comp_len:] if comp_len > 0 else g_logps[:, :0]
            all_logps.append(g_logps)
        if return_entropy:
            return torch.cat(all_logps, dim=0), torch.cat(all_ent, dim=0)
        return torch.cat(all_logps, dim=0)

    def _generate_rollout(self, model, inputs):
        """Generate and cache one rollout; with mu > 1 it is reused for mu policy updates.
        Contains: sampled completions, rewards/advantages, the old (behaviour) logps snapshot, ref logps (KL),
                    cached frozen video/audio embeddings. (Generation and logging happen once per fresh rollout.)"""
        device = self.accelerator.device
        _unwrapped = self.accelerator.unwrap_model(model)

        # -- inputs (video-SALMONN 2+ dataset fields) --
        prompt_ids = inputs["input_ids"].to(device)
        prompt_mask = inputs["attention_mask"].to(device)

        # drop the GT answer, keep only the prompt
        labels = inputs.get("labels", None)
        if labels is not None:
            labels = labels.to(device)
            answer_start = (labels[0] != -100).nonzero(as_tuple=True)[0]
            if len(answer_start) > 0:
                prompt_end_idx = answer_start[0].item()
                prompt_ids = prompt_ids[:, :prompt_end_idx]
                prompt_mask = prompt_mask[:, :prompt_end_idx]

        # [TTI-DBG] check that time markers survive in the prompt (rank 0, first N steps)
        if TTI_DEBUG and self.accelerator.is_main_process and getattr(self, "_tti_dbg_step0", None) is None:
            self._tti_dbg_step0 = self.state.global_step
        if (TTI_DEBUG and self.accelerator.is_main_process
                and self.state.global_step < self._tti_dbg_step0 + TTI_DEBUG_STEPS):
            _id_range = getattr(_unwrapped.config, "time_token_id_range", None)
            _n_mark = count_time_markers(prompt_ids[0], _id_range)
            _tti_dbg(f"⑤ step {self.state.global_step}: prompt_len={prompt_ids.size(1)}, "
                     f"model.config.time_token_id_range={_id_range}, "
                     f"#time_markers_in_prompt={_n_mark}")
            if _id_range is not None and _n_mark == 0:
                _tti_dbg("WARNING: TTI on (range set) but the prompt contains 0 time markers "
                         "(markers missing from the data? mismatches the rope ON branch)")
            try:
                _vs_id = self.processing_class.convert_tokens_to_ids("<|vision_start|>")
                _row = prompt_ids[0].tolist()
                if _vs_id in _row:
                    _p = _row.index(_vs_id)
                    _dec = self.processing_class.decode(_row[_p:_p + 40], skip_special_tokens=False)
                    _tti_dbg(f"   prompt[vision_start:+40]= {_dec!r}")
            except Exception as _e:
                _tti_dbg(f"   failed to decode the vision_start slice: {_e}")

        # multimodal inputs
        pixel_values_videos = inputs.get("pixel_values_videos", None)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(device=device, dtype=torch.bfloat16)
        video_grid_thw = inputs.get("video_grid_thw", None)
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(device)
        audio_feature = inputs.get("audio_feature", None)
        if audio_feature is not None:
            audio_feature = audio_feature.to(device=device, dtype=torch.bfloat16)
        audio_lengths = inputs.get("audio_lengths", None)
        second_per_grid_ts = inputs.get("second_per_grid_ts", None)
        if second_per_grid_ts is not None:
            second_per_grid_ts = second_per_grid_ts.to(device)

        # GT for the rewards: time segments parsed from the gpt answer
        from reward_functions import decode_vtg_time
        gt_intervals = []
        raw_labels = inputs.get("labels", None)
        if raw_labels is not None:
            raw_labels = raw_labels.to(device)
            gt_token_ids = raw_labels[0][raw_labels[0] != -100]
            if len(gt_token_ids) > 0:
                gt_answer = self.processing_class.decode(gt_token_ids, skip_special_tokens=False)
                segments = re.findall(
                    r"[Ff]rom\s+((?:<t\d>)+<tdot><t\d>)\s+to\s+((?:<t\d>)+<tdot><t\d>)",
                    gt_answer
                )
                for start_str, end_str in segments:
                    s = decode_vtg_time(start_str)
                    e = decode_vtg_time(end_str)
                    if s is not None and e is not None and e > s:
                        gt_intervals.append((s, e))

        # ── Generate completions ──
        all_completion_ids = []
        prompt_length = prompt_ids.size(1)
        # Disable gradient checkpointing during generate (iterate modules directly, bypassing wrappers).
        for _m in model.modules():
            if hasattr(_m, "gradient_checkpointing"):
                _m.gradient_checkpointing = False

        gen_kwargs = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": 1.0,
            "top_k": 50,
            "repetition_penalty": 1.0,
        }
        if pixel_values_videos is not None:
            gen_kwargs["pixel_values_videos"] = pixel_values_videos
        if video_grid_thw is not None:
            gen_kwargs["video_grid_thw"] = video_grid_thw
        if audio_feature is not None:
            gen_kwargs["audio_feature"] = audio_feature
        if audio_lengths is not None:
            gen_kwargs["audio_lengths"] = audio_lengths

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            raw_model = (
                unwrapped_model.get_base_model()
                if hasattr(unwrapped_model, "get_base_model")
                else unwrapped_model
            )
            if self.accelerator.is_main_process and self.state.global_step == 0:
                gc = raw_model.generation_config
                print(f"[GDPO DEBUG] gen_kwargs (sampling): do_sample={gen_kwargs.get('do_sample')}, "
                      f"temperature={gen_kwargs.get('temperature')}, top_p={gen_kwargs.get('top_p')}, "
                      f"top_k={gen_kwargs.get('top_k', '<not set>')}, "
                      f"repetition_penalty={gen_kwargs.get('repetition_penalty', '<not set>')}")
                print(f"[GDPO DEBUG] raw_model.generation_config: "
                      f"do_sample={getattr(gc, 'do_sample', None)}, "
                      f"temperature={getattr(gc, 'temperature', None)}, "
                      f"top_p={getattr(gc, 'top_p', None)}, top_k={getattr(gc, 'top_k', None)}, "
                      f"repetition_penalty={getattr(gc, 'repetition_penalty', None)}")
            for _ in range(self.num_generations):
                gen_ids = raw_model.generate(**gen_kwargs)
                gen_ids = gen_ids[:, prompt_length:]
                all_completion_ids.append(gen_ids)

        # generate done: re-enable gradient checkpointing (use_reentrant=False)
        import functools as _functools
        from torch.utils.checkpoint import checkpoint as _torch_checkpoint
        _gc_func = _functools.partial(_torch_checkpoint, use_reentrant=False)
        for _m in model.modules():
            if hasattr(_m, "gradient_checkpointing"):
                _m.gradient_checkpointing = True
                _m._gradient_checkpointing_func = _gc_func

        # pad and concatenate
        import torch.nn.functional as F
        max_len = max(c.size(1) for c in all_completion_ids)
        pad_id = self.processing_class.pad_token_id or 0
        padded = [F.pad(c, (0, max_len - c.size(1)), value=pad_id) for c in all_completion_ids]
        completion_ids = torch.cat(padded, dim=0)
        prompt_ids = prompt_ids.repeat(self.num_generations, 1)
        prompt_mask = prompt_mask.repeat(self.num_generations, 1)

        # EOS mask
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        if completion_ids.size(1) > 0 and is_eos.any():
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        actual_lengths = completion_mask.sum(dim=1).tolist()
        if self.accelerator.is_main_process:
            print(f"[GDPO STEP] prompt_len={prompt_length}, actual_lengths={[int(l) for l in actual_lengths]}, comp_len={completion_ids.size(1)}")
        comp_len = completion_ids.size(1)

        # Remove video/audio placeholder tokens from the completion (avoids feature-count mismatch in logprobs)
        _VIDEO_TOKEN_ID = 151656
        completion_ids_clean = completion_ids.clone()
        completion_ids_clean[completion_ids_clean == _VIDEO_TOKEN_ID] = pad_id
        if hasattr(_unwrapped.config, 'audio_token_id'):
            completion_ids_clean[completion_ids_clean == _unwrapped.config.audio_token_id] = pad_id

        prompt_ids_single = prompt_ids[:1]
        prompt_mask_single = prompt_mask[:1]

        # -- encoding cache (frozen video/audio encoders, once per rollout) --
        raw_encode_model = _unwrapped.get_base_model() if hasattr(_unwrapped, "get_base_model") else _unwrapped
        cached_video_embeds = None
        cached_audio_embeds = None
        with torch.no_grad():
            if pixel_values_videos is not None:
                pv = pixel_values_videos.type(raw_encode_model.visual.dtype)
                cached_video_embeds = raw_encode_model.visual(pv, grid_thw=video_grid_thw)
            if audio_feature is not None:
                af = audio_feature.type(raw_encode_model.audio.dtype)
                cached_audio_embeds = raw_encode_model.audio(af).flatten(0, 1)
        audio_token_id_cfg = getattr(_unwrapped.config, "audio_token_id", None)

        # rollout state cache (reused as is by the next mu-1 compute_loss calls)
        R = {
            "prompt_ids_single": prompt_ids_single,
            "prompt_mask_single": prompt_mask_single,
            "completion_ids_clean": completion_ids_clean,
            "completion_mask": completion_mask,
            "comp_len": comp_len,
            "prompt_length": prompt_length,
            "video_grid_thw": video_grid_thw,
            "audio_lengths": audio_lengths,
            "second_per_grid_ts": second_per_grid_ts,
            "cached_video_embeds": cached_video_embeds,
            "cached_audio_embeds": cached_audio_embeds,
            "audio_token_id_cfg": audio_token_id_cfg,
        }

        # -- old (behaviour) policy logps snapshot --
        # mu=1 regenerates every step, so old = current policy detached (no extra forward).
        # mu>1 takes a separate no_grad snapshot that stays fixed as the ratio reference for the reuse steps.
        if self.num_iterations > 1:
            with torch.no_grad():
                R["old_per_token_logps"] = self._completion_logps(model, R, raw_encode_model)
        else:
            R["old_per_token_logps"] = None  # compute_loss uses per_token_logps.detach()

        # -- reference logps (for KL; fixed regardless of the policy, so cached) --
        R["ref_per_token_logps"] = None
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    R["ref_per_token_logps"] = self._completion_logps(self.ref_model, R, raw_encode_model)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        R["ref_per_token_logps"] = self._completion_logps(model, R, raw_encode_model)

        # ── Decode completions ──
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)
        _TIME_TOKENS = {f"<t{i}>" for i in range(10)} | {"<tdot>"}
        _special_to_remove = set(self.processing_class.all_special_tokens) - _TIME_TOKENS
        for tok in _special_to_remove:
            completions = [c.replace(tok, "") for c in completions]
        completions = [re.sub(r"<\|im_start\|>\s*\w+\s*", "", c).strip() for c in completions]
        if self.accelerator.is_main_process:
            print(f"[GDPO SAMPLE] completion[0]: {completions[0]}")

        # ── Rewards + advantages ──
        rewards_per_func = torch.zeros(len(completions), len(self.reward_funcs), device=device)
        gt_intervals_repeated = [gt_intervals] * self.num_generations
        for i, reward_func in enumerate(self.reward_funcs):
            output = reward_func(completions=completions, gt_intervals=gt_intervals_repeated)
            rewards_per_func[:, i] = torch.tensor(output, dtype=torch.float32, device=device)
        rewards = rewards_per_func.sum(dim=1)
        advantages = self._compute_gdpo_advantages(rewards_per_func, rewards)
        # For multi-segment (GT >= 2) prompts, scale the advantage by w to offset the single-segment prior.
        #   All rollouts of a step share the same GT, so the whole group gets the same scale
        #   (applied after normalization: relative signal kept, only the gradient size scales by w).
        _n_gt = len(gt_intervals)
        _is_multi = _n_gt >= 2
        if _is_multi and self.multi_seg_weight != 1.0:
            advantages = advantages * self.multi_seg_weight
        if self.accelerator.is_main_process:
            print(f"[GDPO sep2] n_gt={_n_gt} multi={_is_multi} "
                  f"adv_weight={self.multi_seg_weight if _is_multi else 1.0}")
        R["rewards_per_func"] = rewards_per_func
        R["rewards"] = rewards
        R["advantages"] = advantages
        # [dataset-tag] source of this rollout (one step = one prompt -> one tag), for per-dataset reward logging.
        _vid = inputs.get("video", None)
        _vp = _vid[0] if isinstance(_vid, (list, tuple)) and _vid else _vid
        R["source_tag"] = _dataset_tag(_vp)

        # -- debug: raw and second-level GT / predicted segments (rank 0) --
        if self.accelerator.is_main_process:
            _SEG_CAPTURE_DEBUG = re.compile(
                r"[Ff]rom\s+((?:<t\d>){1,4}<tdot><t\d>)\s+to\s+((?:<t\d>){1,4}<tdot><t\d>)"
            )

            def _answer_tuples(text):
                m = re.search(r"<answer>(.*?)</answer>", text, re.S)
                ans = m.group(1) if m else text
                segs = []
                for s_str, e_str in _SEG_CAPTURE_DEBUG.findall(ans):
                    s = decode_vtg_time(s_str)
                    e = decode_vtg_time(e_str)
                    if s is not None and e is not None:
                        segs.append(f"({s:.1f}, {e:.1f})")
                return ", ".join(segs) if segs else None   # None = answer could not be parsed

            gt_raw_str = "[none]"
            if raw_labels is not None:
                _gt_ids = raw_labels[0][raw_labels[0] != -100]
                if len(_gt_ids) > 0:
                    gt_answer_raw = self.processing_class.decode(_gt_ids, skip_special_tokens=False)
                    gt_raw_segs = [f"from {s} to {e}" for s, e in _SEG_CAPTURE_DEBUG.findall(gt_answer_raw)]
                    if gt_raw_segs:
                        gt_raw_str = ", ".join(gt_raw_segs)
            gt_sec_str = ", ".join(f"({s:.1f}, {e:.1f})" for s, e in gt_intervals) if gt_intervals else "[none]"
            print(f"[GDPO SAMPLES] GT raw: {gt_raw_str}")
            print(f"[GDPO SAMPLES] GT sec: {gt_sec_str}")
            # Channel names come from the reward functions' __name__ (set by load_reward_module), no hardcoded indices,
            # so any channel layout is printed correctly.
            _chan_names = [getattr(f, "__name__", f"ch{j}") for j, f in enumerate(self.reward_funcs)]
            for gi, c in enumerate(completions):
                _vals = ", ".join(f"{nm}={rewards_per_func[gi, j].item():.3f}"
                                  for j, nm in enumerate(_chan_names))
                _ans = _answer_tuples(c)
                print(f"  [{gi}] {_vals}")
                print(f"       raw:    {c}")
                if _ans is not None:
                    print(f"       answer: {_ans}")               # (s, e) tuples
                else:
                    print(f"       answer: [parse fail]")          # parse failed (see raw above)

        return R

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GDPOTrainer does not support return_outputs=True")

        _unwrapped = self.accelerator.unwrap_model(model)

        # -- prompt-signature rollout cache (order independent, DDP safe) --
        #   When the same prompt arrives mu times, generate on the first call and reuse it mu-1 times.
        #   The cache key is the prompt signature, so scattered microbatch order (block repeat / DDP)
        #   cannot cause wrong reuse or crashes (mismatch = miss -> regenerate). Entries reaching mu are evicted.
        sig = self._prompt_sig(inputs)
        entry = self._rollout_cache.get(sig)
        if entry is None or entry["uses"] >= self.num_iterations:
            # fresh: generate a new rollout (old_logps snapshot fixed here). A cache miss (mu_iter >= 1) also self-heals here.
            R = self._generate_rollout(model, inputs)
            entry = {"R": R, "uses": 1}
            self._rollout_cache[sig] = entry
            fresh = True
        else:
            entry["uses"] += 1  # reuse: old_logps stay fixed (same R), only the policy changes -> r != 1
            R = entry["R"]
            fresh = False
        # reached mu -> evict (frees the embedding cache). Cache size stays ~grad_accum.
        if entry["uses"] >= self.num_iterations:
            self._rollout_cache.pop(sig, None)
        # If the cache grows abnormally (order disturbed), drop the oldest entries (memory guard)
        while len(self._rollout_cache) > self.grad_accum + 1:
            self._rollout_cache.pop(next(iter(self._rollout_cache)))
        if self.accelerator.is_main_process and self.num_iterations > 1:
            print(f"[GDPO ROLLOUT] step={self.state.global_step} "
                  f"uses={entry['uses']}/{self.num_iterations} ({'fresh' if fresh else 'cached'}) "
                  f"cache={len(self._rollout_cache)}")

        comp_len = R["comp_len"]
        completion_mask = R["completion_mask"]
        advantages = R["advantages"]

        # -- new policy per-token logps (with grad), recomputed for the cached completions --
        #    return_entropy=True would also return the policy entropy (logging only, detached).
        raw_encode_model = _unwrapped.get_base_model() if hasattr(_unwrapped, "get_base_model") else _unwrapped
        per_token_logps = self._completion_logps(
            model, R, raw_encode_model, return_entropy=False)  # entropy off (saves memory)

        # old (behaviour) logps: generation-time snapshot for mu>1, current policy detached for mu=1 (ratio 1).
        old_per_token_logps = R["old_per_token_logps"]
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        # -- KL (cached ref vs new policy) --
        per_token_kl = None
        if self.beta != 0.0 and R["ref_per_token_logps"] is not None:
            ref_per_token_logps = R["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )

        # ── Loss ──
        if self.clip_higher:
            # DAPO clipped surrogate: -min( r·A, clip(r, 1-ε_low, 1+ε_high)·A )
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if per_token_kl is not None:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            comp_lengths = completion_mask.sum(dim=1).clamp(min=1)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / comp_lengths).mean()
        elif self.use_grpo:
            per_token_loss = torch.exp(per_token_logps - old_per_token_logps) * advantages.unsqueeze(1)
            if per_token_kl is not None:
                per_token_loss = -(per_token_loss - self.beta * per_token_kl)
            else:
                per_token_loss = -per_token_loss
            comp_lengths = completion_mask.sum(dim=1).clamp(min=1)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / comp_lengths).mean()
        else:
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if per_token_kl is not None:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # -- logging --
        rewards_per_func = R["rewards_per_func"]
        rewards = R["rewards"]
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        # [wandb] advantage statistics (values actually used, after group norm and multi_seg_weight)
        with torch.no_grad():
            self._metrics["advantage_mean"].append(self.accelerator.gather_for_metrics(advantages).mean().item())
            self._metrics["advantage_std"].append(self.accelerator.gather_for_metrics(advantages).std().item())
            self._metrics["advantage_abs_mean"].append(self.accelerator.gather_for_metrics(advantages.abs()).mean().item())

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            fname = getattr(reward_func, "__name__", f"func_{i}")
            self._metrics[f"rewards/{fname}"].append(reward_per_func[i].item())

        # [dataset-tag] per-dataset reward logging -> W&B keys rewards/<channel>/<tag>.
        #   Each rank has one source per step (same for all rollout rows): build a per-row tag-id tensor,
        #   gather it with the rewards and average per tag. Tags absent from this step are skipped (gap in the plot).
        _rpf = R["rewards_per_func"]
        _sid = DATASET_TAGS.index(R["source_tag"]) if R.get("source_tag") in DATASET_TAGS else -1
        _row_sid = torch.full((_rpf.shape[0],), _sid, device=_rpf.device, dtype=torch.long)
        _g_rpf = self.accelerator.gather_for_metrics(_rpf)
        _g_sid = self.accelerator.gather_for_metrics(_row_sid)
        for _ti, _tname in enumerate(DATASET_TAGS):
            _m = (_g_sid == _ti)
            if _m.any():
                _vals = _g_rpf[_m].mean(0)
                for i, reward_func in enumerate(self.reward_funcs):
                    fname = getattr(reward_func, "__name__", f"func_{i}")
                    self._metrics[f"rewards/{fname}/{_tname}"].append(_vals[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        std_grouped = rewards.view(-1, self.num_generations).std(dim=1)
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped).mean().item())

        # clip-higher diagnostics: mean ratio + clip fraction (to check that the clip binds for mu > 1)
        with torch.no_grad():
            ratio = torch.exp(per_token_logps - old_per_token_logps)
            mask_f = completion_mask.float()
            denom = mask_f.sum().clamp(min=1)
            mean_ratio = (ratio * mask_f).sum() / denom
            clipped = ((ratio > 1.0 + self.epsilon_high) | (ratio < 1.0 - self.epsilon_low)).float()
            clip_frac = (clipped * mask_f).sum() / denom
            self._metrics["ratio"].append(self.accelerator.gather_for_metrics(mean_ratio).mean().item())
            self._metrics["clip_frac"].append(self.accelerator.gather_for_metrics(clip_frac).mean().item())

        if per_token_kl is not None:
            comp_lengths_kl = completion_mask.sum(dim=1).clamp(min=1)
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / comp_lengths_kl).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # The returned loss is the mean for one prompt (over tokens/group) and is not divided by
        # grad_accum: HF Trainer.training_step scales loss /= gradient_accumulation_steps once
        # (model_accepts_loss_kwargs=False), so accumulating N microbatches gives the N-prompt mean.
        # No double scaling.
        return loss

    # ============================================================
    # logging
    # ============================================================

    def log(self, logs: dict, start_time=None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()


# ============================================================
# Model loading (video-SALMONN 2+)
# ============================================================

def load_model_and_tokenizer(model_path, model_base, tti_mode="off"):
    """Load video-SALMONN 2+ (Qwen2.5-VL based).

    tti_mode:
        - "off": ignore the base config's time_token_id_range (overwritten with None) so that
                 rope_index takes the OFF branch. Use when the data has no time markers.
        - "on" : keep the base config. Use when the data contains time markers.
    """
    print(f"[GDPO] Loading VS2+ model (tti_mode={tti_mode})")
    print(f"[GDPO]   model_path (SFT ckpt): {model_path}")
    print(f"[GDPO]   model_base: {model_base}")

    # tokenizer
    tok_path = model_path if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "tokenizer.json")) else model_base
    print(f"[GDPO] Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=10000, padding_side="left")

    # model
    print(f"[GDPO] Loading base model from: {model_base}")
    model = video_SALMONN2_plus.from_pretrained(
        model_base,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )

    # TTI mode: in OFF mode disable the base config's time_token_id_range.
    # The time-token base model has time_token_id_range set; left as is, rope_index would
    # always take the TTI-ON branch and raise IndexError for data without time markers.
    if tti_mode == "off":
        if getattr(model.config, "time_token_id_range", None) is not None:
            print(f"[GDPO] tti_mode=off → clearing model.config.time_token_id_range "
                  f"(was {model.config.time_token_id_range})")
            model.config.time_token_id_range = None
        if getattr(model.config, "time_marker_token_len", None):
            model.config.time_marker_token_len = None
    else:
        # tti_mode == 'on': a merged model (path B) may lack time_token_id_range in its config
        # (lost when merging the SFT model). Restore it from the tokenizer so that it matches the
        # data (special_token markers) and the model's rope also takes the ON (fixed-position) branch
        # during generation. (The time-token base of path A already has it.)
        if getattr(model.config, "time_token_id_range", None) is None:
            t0 = tokenizer.convert_tokens_to_ids("<t0>")
            tdot = tokenizer.convert_tokens_to_ids("<tdot>")
            if isinstance(t0, int) and isinstance(tdot, int) and t0 >= 0 and tdot >= 0:
                model.config.time_token_id_range = (min(t0, tdot), max(t0, tdot))
                model.config.time_marker_token_len = 5
                print(f"[GDPO] tti_mode=on but the config has no time_token_id_range (merged model, path B) "
                      f"-> restored from the tokenizer: {model.config.time_token_id_range}, marker_len=5")
            else:
                print("[GDPO] WARNING: tti_mode=on but the <t0>/<tdot> token ids could not be resolved "
                      "-> time_token_id_range not set (config/data stay out of sync)")
    # [TTI-DBG] final model.config state
    _tti_dbg(f"① load_model: tti_mode={tti_mode} | "
             f"model.config.time_token_id_range={getattr(model.config, 'time_token_id_range', None)} | "
             f"time_marker_token_len={getattr(model.config, 'time_marker_token_len', None)}")
    # Do not call resize_token_embeddings: the base keeps vocab_size=152064 (Qwen padding) to
    # match the SFT adapter's modules_to_save (embed_tokens/lm_head, 152064 rows).
    model = model.to(torch.bfloat16)

    # Load LoRA: audio.layers must be detached (same pattern as SFT train_qwen.py).
    # The adapter_config's base_model_name_or_path is ignored (the loaded model object is passed).
    # modules_to_save=[model.embed_tokens, lm_head] are restored by PeftModel loading,
    # so the time-token embeddings need no manual restore.
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    adapter_loaded = os.path.isdir(model_path) and os.path.exists(adapter_config_path)
    if adapter_loaded:
        # -- (path A) continue the SFT LoRA adapter on top of the base --
        print(f"[GDPO] Loading LoRA adapter: {model_path}")
        audio_layers = model.audio.layers
        del model.audio.layers
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
        model.model.audio.layers = audio_layers

        # -- restrict RL trainable modules (path A only) --
        # SFT adapters load LoRA(q,k,v,o) + visual.merger + audio.qformer/q_tokens/audio_proj +
        # embed_tokens + lm_head as trainable. RL uses a single LR and cannot reproduce the
        # per-module SFT LRs, and noisy reward gradients on the aligners risk damaging the
        # multimodal grounding learned in SFT. So training is restricted to the policy:
        #   trained: LoRA(q,k,v) + embed_tokens + lm_head
        #   freeze: o_proj LoRA, visual.merger, audio.qformer/q_tokens/audio_proj
        _KEEP_LORA = ("q_proj", "k_proj", "v_proj")  # o_proj excluded
        _KEEP_SAVE = ("embed_tokens", "lm_head")  # merger / audio.* excluded
        _kept_names, _frozen_names = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            keep = any(m in n for m in _KEEP_LORA) if "lora_" in n \
                else any(m in n for m in _KEEP_SAVE)
            if keep:
                _kept_names.append(n)
            else:
                p.requires_grad_(False)
                _frozen_names.append(n)

        def _mod_summary(names):
            tags = ("q_proj", "k_proj", "v_proj", "o_proj", "visual.merger",
                    "audio.qformer", "audio.q_tokens", "audio.audio_proj",
                    "embed_tokens", "lm_head")
            return sorted({m for n in names for m in tags if m in n})

        _n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[GDPO] RL trainable modules restricted -> trained: {_mod_summary(_kept_names)}")
        print(f"[GDPO]                                   frozen:  {_mod_summary(_frozen_names)}")
        print(f"[GDPO] trainable params: {_n_train/1e6:.1f}M")
        # ─────────────────────────────────────────────────────────────────
    else:
        # -- (path B) fresh LoRA: keep the merged model and add a new LoRA from config.lora in main().
        #    get_peft_model freezes all non-LoRA parameters automatically, so no extra
        #    restriction is needed here (embed/lm_head are frozen too). --
        print("[GDPO] No LoRA adapter found -> fresh RL LoRA path "
              "(a new LoRA from config.lora is applied in main)")

    # With gradient checkpointing and a frozen base, the frozen embedding outputs
    # need requires_grad=True for gradients to flow.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Bypass HF generate's _validate_model_kwargs (monkey patch).
    # With the PEFT (LoraModel) wrapper HF only sees `*args, **kwargs` when inspecting the
    # forward signature and wrongly reports pixel_values_videos / video_grid_thw /
    # audio_feature / audio_lengths as "unused". base_model.forward /
    # prepare_inputs_for_generation handle them correctly.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base._validate_model_kwargs = lambda *a, **kw: None

    # attach the tokenizer
    if not hasattr(model, "tokenizer"):
        model.tokenizer = tokenizer

    print(f"[GDPO] VS2+ model loaded successfully")
    return model, tokenizer


# ============================================================
# Config loading
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    def _env_replace(match):
        return os.environ.get(match.group(1), match.group(0))
    raw = re.sub(r"\$\{(\w+)\}", _env_replace, raw)
    return yaml.safe_load(raw)


# ============================================================
# Reward functions
# ============================================================

def make_reward_functions(specs):
    """Spec list [(name, fn, needs_gt), ...] from load_reward_module -> callables for the trainer.
    needs_gt=False (format) is called with the completion only, True with (completion, gt).
    Works for any number of channels."""
    out = []
    for name, fn, needs_gt in specs:
        if needs_gt:
            def _r(completions, gt_intervals=None, _fn=fn, **kwargs):
                gt = gt_intervals or [[] for _ in completions]
                return [_fn(c, g) for c, g in zip(completions, gt)]
        else:
            def _r(completions, _fn=fn, **kwargs):
                return [_fn(c) for c in completions]
        _r.__name__ = name
        out.append(_r)
    return out


# ============================================================
# [VAL] validation-set evaluation inside the trainer (no reload), optional
#
# For checkpoint selection; only active with val.enabled (or --val_path), otherwise no change.
# At every save step (on_save), greedy-generate the val prompts with the current policy ->
#   sample mIoU + pairwise F1_avg (θ = 0.1/0.3/0.5/0.7/0.9) -> their mean (combined).
# The metric definitions replicate eval/eval_miou.py.
# Appended per step to val_metrics.jsonl and logged to W&B.
# ============================================================
VAL_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
# Time-token "From <t..> to <t..>" parsing (same regex/decoding as eval_miou.py)
_VAL_TOKTIME = r"(?:<t\d>)+(?:<tdot>(?:<t\d>)+)?"
_VAL_TOK_SEG = re.compile(rf"({_VAL_TOKTIME})\s*(?:to|-|–|—|~)\s*({_VAL_TOKTIME})", re.IGNORECASE)


def _val_fix(s, e, max_time):
    if e <= s:
        e = min(s + 0.1, max_time)
    return [min(s, max_time), min(e, max_time)]


def _val_decode_tok(token_str, max_time):
    if "<tdot>" in token_str:
        a, _, b = token_str.partition("<tdot>")
        ip = re.findall(r"<t(\d)>", a)
        dp = re.findall(r"<t(\d)>", b)
    else:
        ip = re.findall(r"<t(\d)>", token_str)
        dp = []
    if not ip:
        return None
    return min(int("".join(ip)) + (int(dp[0]) / 10.0 if dp else 0.0), max_time)


def _val_parse_tokens(text, max_time):
    out = []
    for sa, sb in _VAL_TOK_SEG.findall(text or ""):
        s, e = _val_decode_tok(sa, max_time), _val_decode_tok(sb, max_time)
        if s is None or e is None:
            continue
        out.append(_val_fix(s, e, max_time))
    return out


def _val_extract_answer_scope(text):
    """CoT: only the content of <answer>...</answer> (the whole text if absent)."""
    spans = re.findall(r"<answer>(.*?)</answer>", text or "", re.DOTALL | re.IGNORECASE)
    return " ".join(spans) if spans else (text or "")


def _val_tiou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _val_best_iou(seg, others):
    return max((_val_tiou(seg, o) for o in others), default=0.0)


def _val_merge(segs):
    """Merge overlapping intervals into a disjoint union."""
    if not segs:
        return []
    s = sorted([[float(a), float(b)] for a, b in segs], key=lambda x: x[0])
    out = [s[0][:]]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _val_total(segs):
    return sum(e - s for s, e in segs)


def _val_sample_iou(gt, pred):
    """One IoU per sample (R-AVST all_iou): merged GT vs merged pred, inter/union."""
    g, p = _val_merge(gt), _val_merge(pred)
    if not g and not p:
        return 0.0
    inter = 0.0
    for gs, ge in g:
        for ps, pe in p:
            inter += max(0.0, min(ge, pe) - max(gs, ps))
    union = _val_total(g) + _val_total(p) - inter
    return inter / union if union > 0 else 0.0


def _val_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _val_frac_ge(arr, th):
    return _val_mean([1.0 if x >= th else 0.0 for x in arr]) if arr else 0.0


def compute_val_metrics(records):
    """records: [{"gt": [[s,e]..], "pred": [[s,e]..]}] -> same definitions as eval_miou.py.

    To match the test metrics, IoU uses the sample method and F1 the pairwise method.
        sample_miou  = 100 * mean_over_samples( all_iou(merge(gt), merge(pred)) )   <- eval_miou sample block
        pairwise_miou= 100 * mean_over_all_GT_segs( best_iou(g, pred) )             <- (reference)
        f1_avg       = mean_over_θ( 2PR/(P+R) ),  R/P = pairwise best_iou>=θ frac   <- eval_miou pairwise block
        combined     = (sample_miou + f1_avg) / 2          <- checkpoint selection criterion

    Key names seg_miou/f1_avg/combined are kept for compatibility; seg_miou holds the
          sample IoU (same as sample_miou). The pairwise value is kept as pairwise_miou.
    """
    gt_iou_all, pred_iou_all, sample_ious = [], [], []
    for r in records:
        gt, pred = r["gt"], r["pred"]
        if not gt:
            continue
        gt_iou_all.extend(_val_best_iou(g, pred) for g in gt)
        pred_iou_all.extend(_val_best_iou(p, gt) for p in pred)
        sample_ious.append(_val_sample_iou(gt, pred))
    sample_miou = 100.0 * _val_mean(sample_ious)
    pairwise_miou = 100.0 * _val_mean(gt_iou_all)
    f1, rec, prc = {}, {}, {}
    for t in VAL_THRESHOLDS:
        R = _val_frac_ge(gt_iou_all, t)
        P = _val_frac_ge(pred_iou_all, t)
        rec[str(t)] = round(100.0 * R, 4)
        prc[str(t)] = round(100.0 * P, 4)
        f1[str(t)] = round(100.0 * (2 * P * R / (P + R) if (P + R) > 0 else 0.0), 4)
    f1_avg = _val_mean(list(f1.values()))
    n_parse_ok = sum(1 for r in records if r["pred"])
    return {
        "seg_miou": round(sample_miou, 4),      # sample IoU (compatibility key)
        "sample_miou": round(sample_miou, 4),
        "pairwise_miou": round(pairwise_miou, 4),
        "f1_avg": round(f1_avg, 4),
        "combined": round((sample_miou + f1_avg) / 2.0, 4),
        "R": rec, "P": prc, "F1": f1,
        "n_samples": len(records),
        "n_gt_segs": len(gt_iou_all),
        "n_pred_segs": len(pred_iou_all),
        "n_parse_ok": n_parse_ok,
    }


@torch.no_grad()
def _val_greedy_generate(trainer, inputs, max_new_tokens):
    """Greedy-generate one sample with the same multimodal inputs as training -> completion text (special tokens kept)."""
    model = trainer.model
    acc = trainer.accelerator
    device = acc.device
    tok = trainer.processing_class

    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    labels = inputs.get("labels", None)
    if labels is not None:
        labels = labels.to(device)
        answer_start = (labels[0] != -100).nonzero(as_tuple=True)[0]
        if len(answer_start) > 0:
            pe = answer_start[0].item()
            prompt_ids = prompt_ids[:, :pe]
            prompt_mask = prompt_mask[:, :pe]

    pvv = inputs.get("pixel_values_videos", None)
    if pvv is not None:
        pvv = pvv.to(device=device, dtype=torch.bfloat16)
    vgt = inputs.get("video_grid_thw", None)
    if vgt is not None:
        vgt = vgt.to(device)
    af = inputs.get("audio_feature", None)
    if af is not None:
        af = af.to(device=device, dtype=torch.bfloat16)
    al = inputs.get("audio_lengths", None)
    # Video time-axis RoPE scale. Without it time-token grounding is misaligned and the mIoU
    #   drops to about half compared with the eval inference path (train_qwen.py passes **inputs).
    spg = inputs.get("second_per_grid_ts", None)
    if spg is not None and hasattr(spg, "to"):
        spg = spg.to(device)

    prompt_length = prompt_ids.size(1)
    # Disable gradient checkpointing during generate (same pattern as the rollout)
    for _m in model.modules():
        if hasattr(_m, "gradient_checkpointing"):
            _m.gradient_checkpointing = False

    gen_kwargs = {
        "input_ids": prompt_ids,
        "attention_mask": prompt_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,        # greedy (deterministic) for reproducible selection
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    if pvv is not None:
        gen_kwargs["pixel_values_videos"] = pvv
    if vgt is not None:
        gen_kwargs["video_grid_thw"] = vgt
    if af is not None:
        gen_kwargs["audio_feature"] = af
    if al is not None:
        gen_kwargs["audio_lengths"] = al
    if spg is not None:
        gen_kwargs["second_per_grid_ts"] = spg

    import warnings as _warnings
    with unwrap_model_for_generation(model, acc) as unwrapped_model:
        raw_model = (
            unwrapped_model.get_base_model()
            if hasattr(unwrapped_model, "get_base_model")
            else unwrapped_model
        )
        # Greedy (do_sample=False) but model.generation_config still has temperature/top_p/top_k,
        # which warns on every sample (harmless but noisy). Suppress only that warning.
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", message=r"`do_sample` is set to `False`")
            gen_ids = raw_model.generate(**gen_kwargs)
    gen_ids = gen_ids[:, prompt_length:]

    # restore gradient checkpointing (use_reentrant=False)
    import functools as _ft
    from torch.utils.checkpoint import checkpoint as _ckpt
    _gcf = _ft.partial(_ckpt, use_reentrant=False)
    for _m in model.modules():
        if hasattr(_m, "gradient_checkpointing"):
            _m.gradient_checkpointing = True
            _m._gradient_checkpointing_func = _gcf

    return tok.batch_decode(gen_ids, skip_special_tokens=False)[0]


class ValEvalCallback(transformers.TrainerCallback):
    """Greedy evaluation on several val sets at every save step -> val_metrics.jsonl + W&B.

    Each set is logged separately (val_<name>/*); the mean of the per-set metrics is stored
    as top-level combined/seg_miou/f1_avg (one set: the same as before).
    Training is never stopped.
    """

    def __init__(self, trainer, val_sets, out_dir,
                 max_new_tokens=256, max_time=999.9, natural=False):
        # val_sets: list of (name, LazySupervisedDataset)
        self.trainer = trainer
        self.out_dir = out_dir
        self.max_new_tokens = int(max_new_tokens)
        self.max_time = float(max_time)
        self.natural = bool(natural)
        self.jsonl_path = os.path.join(out_dir, "val_metrics.jsonl")
        # GT of each set: LazySupervisedDataset shuffles list_data_dict (dataset.py), so read
        # gt_segments in the shuffled order to keep predictions and GT aligned.
        self.val_sets = []
        for name, ds in val_sets:
            gt_list = []
            for x in ds.list_data_dict:
                segs = x.get("gt_segments") or []
                gt_list.append([[float(s), float(e)] for s, e in segs])
            assert len(gt_list) == len(ds), (
                f"val[{name}] GT({len(gt_list)}) != dataset({len(ds)}): order/content mismatch")
            self.val_sets.append({"name": name, "ds": ds, "gt": gt_list})

    def on_save(self, args, state, control, **kwargs):
        self._run(state)

    def _eval_one(self, ds, gt_list, acc, world, rank):
        """Sharded greedy evaluation of one val set -> metric dict (main process) / None (others)."""
        from accelerate.utils import gather_object
        trainer = self.trainer
        local = []
        for i in range(len(ds)):
            if i % world != rank:
                continue
            inst = ds[i]
            batch = trainer.data_collator([inst])
            text = _val_greedy_generate(trainer, batch, self.max_new_tokens)
            scope = _val_extract_answer_scope(text) if self.natural else text
            pred = _val_parse_tokens(scope, self.max_time)
            local.append({"idx": i, "gt": gt_list[i], "pred": pred})
        gathered = gather_object(local)   # collective: every rank must call it
        if not acc.is_main_process:
            return None
        seen, recs = set(), []
        for r in gathered:
            if r["idx"] in seen:
                continue
            seen.add(r["idx"])
            recs.append(r)
        return compute_val_metrics(recs)

    def _run(self, state):
        trainer = self.trainer
        acc = trainer.accelerator
        world = max(1, int(getattr(acc, "num_processes", 1)))
        rank = int(getattr(acc, "process_index", 0))
        step = int(state.global_step)

        model = trainer.model
        was_training = model.training
        model.eval()
        per_set = {}   # name -> metric dict (main process only)
        try:
            for vs in self.val_sets:
                m = self._eval_one(vs["ds"], vs["gt"], acc, world, rank)
                if m is not None:
                    per_set[vs["name"]] = m
        finally:
            if was_training:
                model.train()

        if not acc.is_main_process:
            return

        names = [vs["name"] for vs in self.val_sets]

        def _avg(key):
            return sum(per_set[n][key] for n in names) / len(names)

        # top level = mean over sets
        rec = {
            "step": step,
            "combined": round(_avg("combined"), 4),
            "seg_miou": round(_avg("seg_miou"), 4),
            "f1_avg": round(_avg("f1_avg"), 4),
            "n_parse_ok": sum(per_set[n]["n_parse_ok"] for n in names),
            "n_samples": sum(per_set[n]["n_samples"] for n in names),
            "sets": per_set,
        }
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        parts = "  ||  ".join(
            f"{n}: comb={per_set[n]['combined']:.2f}"
            f"(mIoU={per_set[n]['sample_miou']:.2f},F1={per_set[n]['f1_avg']:.2f},"
            f"parse_ok={per_set[n]['n_parse_ok']})"
            for n in names)
        print(f"[VAL] step {step}: AVG combined={rec['combined']:.2f} "
              f"(seg_miou={rec['seg_miou']:.2f}, f1_avg={rec['f1_avg']:.2f})  ||  {parts}")

        log = {
            "val/combined": rec["combined"],
            "val/seg_miou": rec["seg_miou"],
            "val/f1_avg": rec["f1_avg"],
        }
        for n in names:
            log[f"val_{n}/combined"] = per_set[n]["combined"]
            log[f"val_{n}/seg_miou"] = per_set[n]["seg_miou"]
            log[f"val_{n}/f1_avg"] = per_set[n]["f1_avg"]
            log[f"val_{n}/n_pred_segs"] = per_set[n]["n_pred_segs"]
        try:
            trainer.log(log)
        except Exception as _e:
            print(f"[VAL] wandb log skipped: {_e}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GDPO Training (VS2+)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_base", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="limit the number of steps (e.g. smoke tests). >0 overrides num_train_epochs.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="override the W&B/tracker run name (default: config.logging.run_name).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="gradient accumulation steps (default: config.training.gradient_accumulation_steps). "
                             "Use it to keep the effective batch when the GPU count changes: "
                             "effective batch = grad_accum x num_gpu, and the train sampler's "
                             "block_size is the same product, so 2 GPUs x accum 2 and 1 GPU x accum 4 "
                             "both give effective batch 4 / block_size 4.")
    parser.add_argument("--seed", type=int, default=None,
                        help="training seed (fresh LoRA init + data order + sampling). "
                             "Default: config.training.seed (2024). "
                             "The training data file itself does not change.")
    parser.add_argument("--tti_mode", type=str, default=None, choices=[None, "off", "on"],
                        help="TTI mode. off = ignore time_token_id_range / on = keep the base config. "
                             "Default: config.model.tti_mode or 'off'.")
    parser.add_argument("--reward_module", type=str, default=None,
                        help="reward function module (paper: reward_functions_rM_sep3). "
                             "Default: config.reward.module or reward_functions.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="'True' = resume from the latest checkpoint in output_dir, or a checkpoint path. "
                             "Unset = start from scratch.")
    parser.add_argument("--val_path", type=str, default=None,
                        help="[VAL] validation JSON; evaluated greedily at every save step -> val_metrics.jsonl. "
                             "Default: config.val.dataset_path / val.enabled (off).")
    cli = parser.parse_args()

    cfg = load_config(cli.config) if cli.config else {}

    def _get(cli_val, *keys, default=None):
        if cli_val is not None:
            return cli_val
        d = cfg
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d if d is not None else default

    model_path = _get(cli.model_path, "model", "model_path")
    model_base = _get(cli.model_base, "model", "model_base")
    dataset_path = _get(cli.dataset_path, "data", "dataset_path")
    output_dir = _get(cli.output_dir, "training", "output_dir", default="output/gdpo_vs2plus")

    if model_path is None or dataset_path is None:
        parser.error("--model_path and --dataset_path are required")

    # -- record CLI arguments and environment next to the outputs (like train.log) --
    # rank 0 only; output_dir may not exist yet.
    # Records how the run was launched (arguments + actual command line) for reproduction.
    if _is_rank0():
        os.makedirs(output_dir, exist_ok=True)
        _args_record = {
            "cli_args": vars(cli),                       # parsed argparse arguments
            "argv": sys.argv,                            # actual command line
            "resolved": {                                # final values merged with the config (subset)
                "model_path": model_path,
                "model_base": model_base,
                "dataset_path": dataset_path,
                "output_dir": output_dir,
            },
            "cwd": os.getcwd(),
        }
        # If the file exists (resume / rerun), keep the history by numbering instead of overwriting.
        #   cli_args.json → cli_args.1.json → cli_args.2.json ...
        _args_path = os.path.join(output_dir, "cli_args.json")
        if os.path.exists(_args_path):
            _n = 1
            while os.path.exists(os.path.join(output_dir, f"cli_args.{_n}.json")):
                _n += 1
            _args_path = os.path.join(output_dir, f"cli_args.{_n}.json")
        with open(_args_path, "w", encoding="utf-8") as _f:
            json.dump(_args_record, _f, ensure_ascii=False, indent=2)
        print(f"[GDPO] CLI arguments saved: {_args_path}")

    # GDPO parameters
    num_generations = _get(None, "gdpo", "num_generations", default=8)
    max_completion_length = _get(None, "gdpo", "max_completion_length", default=512)
    beta = _get(None, "gdpo", "beta", default=0.04)
    reward_weights = _get(None, "gdpo", "reward_weights", default=[1.0, 1.0, 1.0])
    # read temperature from config and pass it to GRPOConfig.
    temperature = float(_get(None, "gdpo", "temperature", default=1.0))

    # ── DAPO clip-higher (config: clip.*) ──────────────────────────────────
    #   enabled : use the clipped surrogate path (default behaviour of this trainer).
    #   epsilon / epsilon_high : DAPO values 0.2 / 0.28 (eps_high > eps_low = clip-higher).
    #   num_iterations (mu) : reuse one rollout for mu policy updates. Only mu > 1 gives r != 1,
    #       so the clip binds. mu=1 gives r ~ 1, numerically the same as plain GDPO.
    clip_enabled = bool(_get(None, "clip", "enabled", default=True))
    epsilon_low = float(_get(None, "clip", "epsilon", default=0.2))
    epsilon_high = float(_get(None, "clip", "epsilon_high", default=0.28))
    num_iterations = int(_get(None, "clip", "num_iterations", default=1))

    # Advantage weight for multi-segment (GT >= 2) prompts (config: gdpo.multi_seg_weight)
    #   Default 1.0 (= off). > 1 scales GT >= 2 prompt advantages by w to offset the single-segment prior.
    multi_seg_weight = float(_get(None, "gdpo", "multi_seg_weight", default=1.0))

    # training parameters
    num_epochs = _get(None, "training", "num_train_epochs", default=1)
    batch_size = _get(None, "training", "per_device_train_batch_size", default=1)
    grad_accum = _get(cli.gradient_accumulation_steps,
                      "training", "gradient_accumulation_steps", default=4)
    lr = float(_get(None, "training", "learning_rate", default=5e-6))
    warmup = _get(None, "training", "warmup_ratio", default=0.1)
    scheduler = _get(None, "training", "lr_scheduler_type", default="cosine")
    seed = _get(cli.seed, "training", "seed", default=2024)
    logging_steps = _get(None, "logging", "logging_steps", default=1)
    save_steps = _get(None, "logging", "save_steps", default=500)
    save_total_limit = _get(None, "logging", "save_total_limit", default=3)
    report_to = _get(None, "logging", "report_to", default="tensorboard")
    run_name = _get(cli.run_name, "logging", "run_name", default=os.path.basename(output_dir))

    # TTI mode (cli > config.model.tti_mode > 'off')
    tti_mode = _get(cli.tti_mode, "model", "tti_mode", default="off")
    # tti_mode -> input marker format of the dataset.
    #   on  : interleave special_token markers between video/audio chunks (pairs with the rope ON branch)
    #   off : no markers (pairs with the rope OFF branch)
    # The model config (time_token_id_range) and the dataset (tti_time_format) must agree.
    tti_time_format = "special_token" if tti_mode == "on" else "off"

    # Model
    # Seed before model loading / fresh LoRA init (the Trainer's set_seed runs after get_peft_model, too late).
    # [SEED] log the seed and its source.
    #   The seed only affects the RL stage: fresh LoRA init + data order + sampling.
    #   The SFT model and the training data files do not depend on it.
    print(f"[SEED] training seed = {seed} "
          f"(source={'cli --seed' if cli.seed is not None else 'config.training.seed'})",
          flush=True)
    transformers.set_seed(seed)
    model, tokenizer = load_model_and_tokenizer(model_path, model_base, tti_mode=tti_mode)

    # -- PEFT: fresh RL LoRA (path B) --
    # When model_path is a merged SFT model (no adapter_config.json), a new LoRA is created on
    # the merged weights instead of re-attaching the SFT adapter. GDPOTrainer uses the disabled
    # LoRA as reference (= SFT policy). Only active with config.lora.enabled=true.
    #   For path A (continuing an adapter), set lora.enabled to false
    #   (adding a LoRA to a model that is already a PeftModel would apply it twice).
    peft_config = None
    if _get(None, "lora", "enabled", default=False):
        if isinstance(model, PeftModel):
            print("[GDPO] WARNING: the model already has a LoRA adapter but config.lora.enabled=true "
                  "-> skipping the fresh LoRA (avoids applying it twice); continuing the adapter.")
        else:
            lora_r = int(_get(None, "lora", "r", default=32))
            lora_alpha = int(_get(None, "lora", "lora_alpha", default=64))
            lora_dropout = float(_get(None, "lora", "lora_dropout", default=0.05))
            # LLM block target suffixes; overridable with config lora.target_modules.
            #   Default: attention only (q/k/v/o_proj), no audio/visual modules.
            #   e.g. to include the MLP: target_modules: [q_proj,k_proj,v_proj,o_proj,
            #       gate_proj,up_proj,down_proj] (the paper setting). Limited to model.layers.*.
            _tm_suffixes = tuple(_get(None, "lora", "target_modules",
                                      default=["q_proj", "k_proj", "v_proj", "o_proj"]))
            target_modules = [
                n for n, _ in model.named_modules()
                if n.startswith("model.layers.")
                and n.split(".")[-1] in _tm_suffixes
            ]
            # modules_to_save: embed_tokens / lm_head have separate toggles.
            #   Enabled modules get a full trainable copy (modules_to_save).
            #   The reference (disable_adapter) uses the originals (= merged SFT model), so the KL anchor stays the SFT policy.
            #   Default: both false -> LoRA only.
            train_emb = bool(_get(None, "lora", "train_embeddings", default=False))
            train_lm_head = bool(_get(None, "lora", "train_lm_head", default=False))
            modules_to_save = [m for m, on in (("embed_tokens", train_emb),
                                               ("lm_head", train_lm_head)) if on] or None
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save,
                bias="none",
                task_type="CAUSAL_LM",
            )
            print(f"[GDPO] Fresh RL LoRA: r={lora_r}, alpha={lora_alpha}, "
                  f"#target_modules={len(target_modules)} (suffixes={list(_tm_suffixes)}), "
                  f"modules_to_save={modules_to_save}")

    # Dataset: video-SALMONN 2+ LazySupervisedDataset
    print(f"[GDPO] Loading dataset from {dataset_path}")
    from dataclasses import dataclass
    from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

    from transformers import WhisperFeatureExtractor

    @dataclass
    class GDPODataArgs:
        dataset_use: str = ""
        model_type: str = "qwen2.5vl"
        video_max_frames: int = 128         # SFT setting
        video_min_frames: int = 64          # SFT setting
        base_interval: float = 0.2          # SFT setting (fps 5)
        max_pixels: int = 176400            # same as SFT (28*28*225)
        min_pixels: int = 784               # same as SFT (28*28)
        video_max_frame_pixels: int = 28224 # SFT setting
        video_min_frame_pixels: int = 784   # SFT setting
        video_max_total_pixels: int = 1664 * 28 * 28
        video_min_total_pixels: int = 256 * 28 * 28
        run_test: bool = False
        do_sample: bool = False
        num_sample: int = 1
        train_type: str = "sft"
        # Input-side TTI marker format, derived from tti_mode (on -> special_token, off -> off).
        # LazySupervisedDataset interleaves markers between video/audio chunks with this value.
        # Without it the dataset defaults to 'off' and inserts no markers, mismatching the rope ON branch.
        tti_time_format: str = "special_token"
        feature_size: int = 128
        chunk_length: int = 30
        hop_length: int = 160
        sampling_rate: int = 16000
        image_processor: object = None
        audio_processor: object = None      # required by the SFT dataset code
        # Video decode memo cache size (0 = off). Enabled only for mu > 1 (set below).
        video_cache_size: int = 0

    data_args = GDPODataArgs()
    data_args.dataset_use = dataset_path
    data_args.tti_time_format = tti_time_format
    # With mu > 1 (clip-higher rollout reuse), block repeat requests the same video mu times but
    #   the cached passes discard the decode (wasted decoding, host RAM pressure). Memoize the last
    #   grad_accum+1 videos to skip re-decoding, leaving room for a larger batch.
    #   mu=1 has no reuse, so 0 (off). (Host RAM cost is bounded, tens to ~100+ MB.)
    data_args.video_cache_size = (int(grad_accum) + 1) if int(num_iterations) > 1 else 0
    print(f"[GDPO] tti_mode={tti_mode} → data_args.tti_time_format={tti_time_format}")
    print(f"[GDPO Tier1] video_cache_size={data_args.video_cache_size} "
          f"(mu={num_iterations}, grad_accum={grad_accum}; avoids redundant decoding when mu > 1)")
    # [TTI-DBG] consistency of config (model.time_token_id_range) and data (tti_time_format):
    #   special_token <-> id_range set, off <-> id_range None.
    if TTI_DEBUG and _is_rank0():
        _mc_range = getattr(model.config, "time_token_id_range", None)
        _has_range = (_mc_range is not None)
        _want_markers = (tti_time_format == "special_token")
        _tti_dbg(f"② main: tti_mode={tti_mode}, tti_time_format={tti_time_format}, "
                 f"model.config.time_token_id_range={_mc_range}")
        if _has_range != _want_markers:
            _tti_dbg(f"WARNING desync: tti_time_format={tti_time_format} but "
                     f"model.config.time_token_id_range={_mc_range} "
                     f"(expected ON<->range, OFF<->None); rope branch and markers may not match")
    data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(model_base)
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size,
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )

    dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    print(f"[GDPO] Dataset size: {len(dataset)}")

    # -- [VAL] validation datasets (same data_args as training, only the path differs) --
    #   --val_path forces it on; otherwise config.val.enabled; off when neither is given.
    val_path = _get(cli.val_path, "val", "dataset_path", default=None)
    val_enabled = (cli.val_path is not None) or bool(_get(None, "val", "enabled", default=False))
    val_sets = []   # list of (name, LazySupervisedDataset)

    def _make_val_ds(path):
        vda = GDPODataArgs()
        vda.dataset_use = path
        vda.tti_time_format = tti_time_format
        vda.image_processor = data_args.image_processor
        vda.audio_processor = data_args.audio_processor
        return LazySupervisedDataset(tokenizer=tokenizer, data_args=vda)

    if val_enabled and val_path:
        primary_name = str(_get(None, "val", "name", default="unav"))
        val_sets.append((primary_name, _make_val_ds(val_path)))
        print(f"[GDPO][VAL] val dataset[{primary_name}]: {val_path} (size={len(val_sets[-1][1])})")
        # extra val sets: config.val.extra = [{name, dataset_path}, ...] (optional)
        for ex in (_get(None, "val", "extra", default=None) or []):
            ex_name = str(ex.get("name", f"val{len(val_sets)}"))
            ex_path = ex.get("dataset_path")
            if not ex_path:
                print(f"[GDPO][VAL] WARNING: extra val '{ex_name}' has no dataset_path, skipping")
                continue
            val_sets.append((ex_name, _make_val_ds(ex_path)))
            print(f"[GDPO][VAL] val dataset[{ex_name}]: {ex_path} (size={len(val_sets[-1][1])})")
    elif val_enabled and not val_path:
        print("[GDPO][VAL] WARNING: val.enabled but no val_path; validation disabled")

    # Reward: select the module (one trainer, only the import target changes)
    reward_module = _get(cli.reward_module, "reward", "module", default="reward_functions")
    _specs = load_reward_module(reward_module)
    reward_funcs = make_reward_functions(_specs)
    _chan_names = [n for n, _, _ in _specs]
    print(f"[GDPO] reward_module={reward_module}  channels={_chan_names} "
          f"(iou implementation={getattr(_specs[1][1], '__name__', '?')})")
    # Resize reward_weights to the number of channels
    if len(reward_weights) != len(reward_funcs):
        if len(reward_weights) > len(reward_funcs):
            reward_weights = list(reward_weights)[:len(reward_funcs)]
        else:
            reward_weights = list(reward_weights) + [1.0] * (len(reward_funcs) - len(reward_weights))
        print(f"[GDPO] WARNING: reward_weights resized -> {reward_weights} ({len(reward_funcs)} channels). "
              f"Set reward_weights explicitly to match the reward module.")
    print(f"[GDPO] Reward weights: {reward_weights}")

    # GRPOConfig
    grpo_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup,
        lr_scheduler_type=scheduler,
        beta=beta,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        temperature=temperature,
        bf16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=report_to,
        run_name=run_name,
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,
    )
    # max_steps: cli > config.training.max_steps (applied only when > 0; ignores epochs)
    _max_steps = _get(cli.max_steps, "training", "max_steps", default=None)
    if _max_steps is not None and int(_max_steps) > 0:
        grpo_kwargs["max_steps"] = int(_max_steps)
    grpo_args = GRPOConfig(**grpo_kwargs)
    # Attach clip-higher parameters to args (read via getattr in GDPOTrainer.__init__).
    #   They may not be GRPOConfig constructor fields, so they are set after construction.
    grpo_args.epsilon = epsilon_low
    grpo_args.epsilon_high = epsilon_high
    grpo_args.clip_higher = clip_enabled
    grpo_args.num_iterations = max(1, num_iterations)
    grpo_args.multi_seg_weight = multi_seg_weight   # read via getattr in GDPOTrainer.__init__
    _clip_state = (f"mu={grpo_args.num_iterations} -> clip active"
                   if grpo_args.num_iterations > 1 else "mu=1 -> clip inactive (= plain GDPO)")
    print(f"[GDPO] clip-higher: enabled={clip_enabled}, ε_low={epsilon_low}, ε_high={epsilon_high}, {_clip_state}")
    print(f"[GDPO sep2] multi_seg_weight(w)={multi_seg_weight} "
          f"({'multi-GT advantage x w' if multi_seg_weight != 1.0 else 'w=1 -> off'})")
    # grad_accum support: the block-repeat sampler steps between mu iterations, so the clip
    #   binds correctly with grad_accum > 1.
    _N = int(grad_accum)
    _world = int(os.environ.get("WORLD_SIZE", "1"))
    _eff_batch = _N * _world
    print(f"[GDPO clip2] grad_accum(N)={_N}, world_size={_world} → effective batch(distinct prompt/step)={_eff_batch}")
    print(f"[GDPO clip2] block-repeat sampler: each block of N={_N} prompts is repeated mu={grpo_args.num_iterations} times "
          f"-> one optimizer step per N microbatches (policy updated between mu iterations). "
          f"unique prompts = max_steps x N x world / mu")
    if _eff_batch < 4:
        print(f"[GDPO clip2] WARNING: effective batch={_eff_batch} (<4); reward estimates/gradients may be noisy "
              f"-> increase grad_accum or the number of GPUs.")

    # Trainer (peft_config is passed -> fresh LoRA for path B)
    trainer = GDPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_weights=reward_weights,
        peft_config=peft_config,
    )

    # -- [VAL] attach the greedy evaluation callback at every save step (skipped when val is off) --
    if val_sets:
        val_max_new = int(_get(None, "val", "max_new_tokens", default=max_completion_length))
        val_max_time = float(_get(None, "val", "max_time", default=999.9))
        val_natural = bool(_get(None, "val", "natural", default=False))
        trainer.add_callback(ValEvalCallback(
            trainer=trainer,
            val_sets=val_sets,
            out_dir=output_dir,
            max_new_tokens=val_max_new,
            max_time=val_max_time,
            natural=val_natural,
        ))
        _names = ", ".join(n for n, _ in val_sets)
        print(f"[GDPO][VAL] ValEvalCallback attached: evaluating every save_steps ({save_steps}) "
              f"[{_names}] (greedy, max_new={val_max_new}, max_time={val_max_time}) → "
              f"{output_dir}/val_metrics.jsonl  (top-level combined = mean over sets)")

    # Train (resume: 'True' = latest checkpoint / a path = that checkpoint / None = from scratch)
    _rc = cli.resume_from_checkpoint
    if _rc in ("", "None"):
        _rc = None
    elif _rc == "True":
        _rc = True
    print(f"[GDPO] Starting training... (resume_from_checkpoint={_rc})")
    trainer.train(resume_from_checkpoint=_rc)
    print(f"[GDPO] Saving model to {output_dir}")
    trainer.save_model(output_dir)
    print("[GDPO] Training complete!")


if __name__ == "__main__":
    main()
