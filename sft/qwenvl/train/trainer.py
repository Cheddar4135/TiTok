# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
# Modified by The TiTok Authors (2026) for TiTok; see NOTICE.
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
# Adopted from https://github.com/huggingface/transformers. The original license is located at 'third-party-license/transformers.txt'.

import os
from typing import Dict, List, Optional, Sequence
# from contextlib import contextmanager, nullcontext

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer
from transformers.cache_utils import Cache
from qwenvl.model.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
)
import torch.distributed as dist

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    SaveStrategy,
)

from transformers.trainer_callback import (
    ExportableState,
)

import re
from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOLoss

def _is_peft_model(model):
    # if is_peft_available():
    #     classes_to_check = (PeftModel,) if is_peft_available() else ()
    #     # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
    #     if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
    #         from peft import PeftMixedModel

    #         classes_to_check = (*classes_to_check, PeftMixedModel)
    #     return isinstance(model, classes_to_check)
    return False

class QwenVLTrainer(Trainer):

    def __init__(
        self, 
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dpo_loss_fct = LigerFusedLinearDPOLoss()

    
    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # ---- Per-module LR/WD groups (preferred path) ----
            per_module_overrides = any(
                getattr(self.args, k, None) is not None
                for k in [
                    "lora_lr", "embed_lr", "lm_head_lr",
                    "visual_merger_lr", "audio_qformer_lr",
                    "audio_proj_lr", "audio_q_tokens_lr",
                ]
            )
            if per_module_overrides:
                base_lr = self.args.learning_rate
                base_wd = self.args.weight_decay

                def _matches(name, kind):
                    if kind == "lora":
                        return ("lora_A" in name) or ("lora_B" in name)
                    if kind == "embed":
                        return ".embed_tokens." in name
                    if kind == "lm_head":
                        return name.endswith(".lm_head.weight") or ".lm_head." in name
                    if kind == "visual_merger":
                        return ".visual.merger." in name
                    if kind == "audio_qformer":
                        return ".audio.qformer." in name
                    if kind == "audio_proj":
                        return ".audio.audio_proj." in name
                    if kind == "audio_q_tokens":
                        return name.endswith(".audio.q_tokens") or ".audio.q_tokens" in name
                    return False

                rules = [
                    ("lora",          self.args.lora_lr,          self.args.lora_wd),
                    ("embed",         self.args.embed_lr,         self.args.embed_wd),
                    ("lm_head",       self.args.lm_head_lr,       self.args.lm_head_wd),
                    ("visual_merger", self.args.visual_merger_lr, self.args.visual_merger_wd),
                    ("audio_qformer", self.args.audio_qformer_lr, self.args.audio_qformer_wd),
                    ("audio_proj",    self.args.audio_proj_lr,    self.args.audio_proj_wd),
                    ("audio_q_tokens",self.args.audio_q_tokens_lr,self.args.audio_q_tokens_wd),
                ]

                groups = []
                seen = set()
                for kind, lr, wd in rules:
                    lr_eff = lr if lr is not None else base_lr
                    wd_eff = wd if wd is not None else base_wd
                    decay_params = []
                    nodecay_params = []
                    decay_names = []
                    nodecay_names = []
                    for n, p in opt_model.named_parameters():
                        if not p.requires_grad or id(p) in seen:
                            continue
                        if not _matches(n, kind):
                            continue
                        seen.add(id(p))
                        if (n in decay_parameters) and ("bias" not in n):
                            decay_params.append(p); decay_names.append(n)
                        else:
                            nodecay_params.append(p); nodecay_names.append(n)
                    if decay_params:
                        groups.append({
                            "params": decay_params, "lr": lr_eff, "weight_decay": wd_eff,
                            "name": f"{kind}_decay",
                        })
                    if nodecay_params:
                        groups.append({
                            "params": nodecay_params, "lr": lr_eff, "weight_decay": 0.0,
                            "name": f"{kind}_nodecay",
                        })

                # Catch-all: any other trainable params not matched above (use base lr/wd)
                other_decay = []
                other_nodecay = []
                other_decay_names = []
                other_nodecay_names = []
                for n, p in opt_model.named_parameters():
                    if not p.requires_grad or id(p) in seen:
                        continue
                    seen.add(id(p))
                    if (n in decay_parameters) and ("bias" not in n):
                        other_decay.append(p); other_decay_names.append(n)
                    else:
                        other_nodecay.append(p); other_nodecay_names.append(n)
                if other_decay:
                    groups.append({"params": other_decay, "lr": base_lr,
                                   "weight_decay": base_wd, "name": "other_decay"})
                if other_nodecay:
                    groups.append({"params": other_nodecay, "lr": base_lr,
                                   "weight_decay": 0.0, "name": "other_nodecay"})

                if dist.is_initialized() and dist.get_rank() == 0:
                    print("[OPTIM] Per-module param groups:")
                    for g in groups:
                        n_p = len(g["params"])
                        n_el = sum(p.numel() for p in g["params"])
                        print(f"  {g['name']:<24} lr={g['lr']:.2e}  wd={g['weight_decay']:.3f}  "
                              f"params={n_p}  elems={n_el/1e6:.2f}M")

                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
                # remove default lr/wd so per-group values win
                optimizer_kwargs.pop("lr", None)
                self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
                return self.optimizer

            if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
                projector_parameters = [
                    name for name, _ in opt_model.named_parameters() if "merger" in name
                ]
                if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                    vision_tower_parameters = [
                        name for name, _ in opt_model.named_parameters() if "visual" in name
                    ]
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and n not in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and n in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.vision_tower_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and n not in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and n in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.vision_tower_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
                else:
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def calc_dpo_loss(self, policy_input, policy_target, ref_input, ce_loss=None, beta=0.1):
        lm_head = self.model.lm_head.weight
        dpo_loss, (chosen_logp, reject_logp, chosen_logit, reject_logit, chosen_nll_loss, chosen_rewards, reject_rewards) = self.dpo_loss_fct(lm_head, policy_input, policy_target, ref_input=ref_input, ref_weight=lm_head)
        if ce_loss is not None:
            loss = dpo_loss + beta * ce_loss
        else:
            loss = dpo_loss
        print(f"RANK {dist.get_rank()} chosen: {chosen_rewards.item()}, reject: {reject_rewards.item()}")
        return (loss, dpo_loss, chosen_rewards, reject_rewards)

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        """Override: save BEFORE eval so an OOM during eval doesn't lose the checkpoint.

        Default order is log → eval → save. With v5_b's output_hidden_states=True
        bypassing Liger fused CE, eval forward can OOM on long-sequence T1 samples.
        Save-first preserves the checkpoint regardless of eval outcome.

        Incompatible with save_strategy=BEST (BEST needs eval result first). We use
        save_strategy="steps", so swap is safe.
        """
        # ---- log (identical to parent) ----
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs, start_time)

        # ---- SAVE FIRST (swapped from parent) ----
        if self.control.should_save:
            if self.args.save_strategy == SaveStrategy.BEST:
                raise RuntimeError(
                    "save_strategy='best' is incompatible with this override; "
                    "use 'steps' (save must precede eval to survive eval OOM)."
                )
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        # ---- EVAL AFTER (swapped from parent) ----
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            self._determine_best_metric(metrics=metrics, trial=trial)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        # Bookkeeping fields from the collator that model.forward does not accept
        _ = inputs.pop("source_id", None)
        _ = inputs.pop("task_type", None)

        train_type = inputs.get("train_type", "")
        if train_type == "sft":
            outputs = model(**inputs)
        elif train_type == "dpo":
            policy_input, policy_target = model(**inputs)
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                with torch.no_grad():
                    reference_input, reference_target = model(**inputs)
            outputs = self.calc_dpo_loss(policy_input, policy_target, reference_input)
            
        elif train_type == "gdpo":
            policy_input, policy_target, ce_loss = model(**inputs)
            inputs["train_type"] = "dpo"
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                with torch.no_grad():
                    reference_input, reference_target = model(**inputs)
            outputs = self.calc_dpo_loss(policy_input, policy_target, reference_input, ce_loss=ce_loss)
        else:
            raise NotImplementedError

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        lm_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        loss = lm_loss

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss