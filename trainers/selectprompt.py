

from __future__ import annotations

import copy
import datetime
import os.path as osp
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import AverageMeter, MetricMeter, load_checkpoint, load_pretrained_weights
from utils import OT_PL, curriculum_scheduler, get_masks, output_selected_rate

_tokenizer = _Tokenizer()


def _torch_load_safe(path: str, map_location="cpu"):
    
    try:
        return torch.load(path, map_location=map_location)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        if "weights_only" in str(exc) or "Weights only" in str(exc):
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


def load_clip_to_cpu(cfg):
  
    backbone_name = cfg.MODEL.BACKBONE.NAME
    local_model_path = osp.join(osp.dirname(osp.dirname(__file__)), "clip", f"{backbone_name}.pt")

    if osp.isfile(local_model_path):
        model_path = local_model_path
    else:
        model_path = clip._download(clip._MODELS[backbone_name])

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = _torch_load_safe(model_path, map_location="cpu")

    design_details = {
        "trainer": "SelectPrompt",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):


    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
 

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        pl_cfg = cfg.TRAINER.SELECTPROMPT
        num_classes = len(classnames)
        num_ctx_tokens = pl_cfg.N_CTX
        ctx_init_text = pl_cfg.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"

        if ctx_init_text:
            ctx_init_text = ctx_init_text.replace("_", " ")
            num_ctx_tokens = len(ctx_init_text.split(" "))
            prompt = clip.tokenize(ctx_init_text)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + num_ctx_tokens, :]
            prompt_prefix = ctx_init_text
        else:
            if pl_cfg.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(num_classes, num_ctx_tokens, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(num_ctx_tokens, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * num_ctx_tokens)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context tokens: {num_ctx_tokens}")

        self.ctx = nn.Parameter(ctx_vectors)
        class_names = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in class_names]
        prompts = [prompt_prefix + " " + name + "." for name in class_names]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + num_ctx_tokens :, :])

        self.n_cls = num_classes
        self.n_ctx = num_ctx_tokens
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = pl_cfg.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError(f"Unknown class token position: {self.class_token_position}")

        return prompts

