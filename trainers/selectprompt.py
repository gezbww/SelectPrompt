
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
        pl_cfg = cfg.TRAINER.SelectPROMPT
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


class GeneralizedCrossEntropy(nn.Module):


    def __init__(self, q: float = 1.0) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-6

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, dim=1)
        p = p[torch.arange(p.shape[0], device=p.device), targets]
        p = p.clamp_min(self.epsilon)
        loss = (1.0 - p.pow(self.q)) / self.q
        return loss.mean()


class VisualTokenSelector(nn.Module):

    def __init__(self, cfg, embed_dim: int):
        super().__init__()
        pl_cfg = cfg.TRAINER.SelectPROMPT
        self.embed_dim = int(embed_dim)
        self.keep_ratio = float(pl_cfg.KEEP_RATIO)
        self.min_keep = int(pl_cfg.MIN_KEEP_TOKENS)
        self.max_keep = int(pl_cfg.MAX_KEEP_TOKENS)
        self.topr = int(pl_cfg.TOPR_QUERY_CLASSES)
        self.gumbel_temp = float(pl_cfg.GUMBEL_TEMP)
        self.gumbel_min_temp = float(pl_cfg.GUMBEL_MIN_TEMP)
        self.current_epoch = 0
        self.max_epoch = int(cfg.OPTIM.MAX_EPOCH)

        self.query_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Tanh(),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Tanh(),
        )
        self.score_scale = nn.Parameter(torch.tensor(10.0))

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def _temperature(self) -> float:
        if self.max_epoch <= 1:
            return self.gumbel_min_temp
        progress = min(max(self.current_epoch / float(self.max_epoch - 1), 0.0), 1.0)
        return self.gumbel_temp + progress * (self.gumbel_min_temp - self.gumbel_temp)

    def _get_k(self, num_tokens: int) -> int:
        k = int(round(num_tokens * self.keep_ratio))
        k = max(k, self.min_keep)
        k = min(k, self.max_keep, num_tokens)
        return max(k, 1)

    def _make_query(
        self,
        text_features: torch.Tensor,
        base_logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        mode: str,
    ) -> torch.Tensor:
        prob = torch.softmax(base_logits.detach(), dim=1)
        r = min(self.topr, prob.shape[1])
        top_prob, top_idx = prob.topk(r, dim=1)
        top_prob = top_prob / top_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pseudo_text = text_features.detach()[top_idx]  # [B, R, D]
        pseudo_query = torch.sum(pseudo_text * top_prob.unsqueeze(-1), dim=1)

        if labels is not None and mode == "clean":
            query = text_features.detach()[labels]
        else:
            query = pseudo_query

        return F.normalize(query.float(), dim=-1)

    def _topk_st_mask(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        # scores: [B, N]
        if self.training:
            tau = self._temperature()

            u = torch.rand_like(scores).clamp_(1e-6, 1.0 - 1e-6)
            g = -torch.log(-torch.log(u))
            noisy_scores = (scores + g) / max(tau, 1e-6)
        else:
            tau = max(self.gumbel_min_temp, 1e-6)
            noisy_scores = scores / tau

        topk_idx = noisy_scores.topk(k, dim=1).indices
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, topk_idx, 1.0)


        soft_mask = torch.softmax(scores / tau, dim=1) * float(k)
        return hard_mask - soft_mask.detach() + soft_mask

    def forward(
        self,
        patch_tokens: torch.Tensor,
        text_features: torch.Tensor,
        base_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = "eval",
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        patch_tokens = patch_tokens.float()
        text_features = text_features.float()
        base_logits = base_logits.float()

        bsz, num_tokens, dim = patch_tokens.shape
        k = self._get_k(num_tokens)
        query = self._make_query(text_features, base_logits, labels, mode)


        query_p = F.normalize(self.query_proj(query), dim=-1)
        token_p = F.normalize(self.token_proj(patch_tokens), dim=-1)
        scores = self.score_scale.clamp(1.0, 30.0) * torch.einsum("bd,bnd->bn", query_p, token_p)
        mask = self._topk_st_mask(scores, k)

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        evidence = torch.einsum("bn,bnd->bd", mask / denom, patch_tokens)
        evidence = F.normalize(evidence.float(), dim=-1)

        # Diagnostics/losses.
        selected_ratio = torch.tensor(k / float(num_tokens), device=patch_tokens.device, dtype=torch.float32)
        budget_loss = (selected_ratio - self.keep_ratio) ** 2
        prob = torch.softmax(scores, dim=1).clamp_min(1e-8)
        entropy = -(prob * prob.log()).sum(dim=1).mean() / np.log(float(num_tokens))

        aux = {
            "selected_ratio": selected_ratio.detach(),
            "budget_loss": budget_loss.float(),
            "selection_entropy": entropy.float(),
            "k": torch.tensor(float(k), device=patch_tokens.device),
        }
        return evidence, mask, aux


class ResidualTokenLogitGate(nn.Module):


    def __init__(self, dim: int, hidden_dim: int = 256, dropout: float = 0.1, max_weight: float = 0.4, learnable: bool = True):
        super().__init__()
        self.max_weight = float(max_weight)
        self.learnable = bool(learnable)
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Start from a useful but still safe residual branch.
        nn.init.constant_(self.net[-1].bias, -0.2)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, image_features: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        if not self.learnable:
            return torch.full((image_features.shape[0], 1), self.max_weight, device=image_features.device, dtype=torch.float32)
        x = torch.cat([image_features.float(), evidence.float()], dim=-1)
        return torch.sigmoid(self.net(x)) * self.max_weight


class TextFeatureFiLM(nn.Module):

    def __init__(self, dim: int, hidden_dim: int = 512, dropout: float = 0.1, max_weight: float = 0.1):
        super().__init__()
        self.max_weight = float(max_weight)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gamma = nn.Linear(hidden_dim, dim)
        self.beta = nn.Linear(hidden_dim, dim)
        self.gate = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, text_features: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        h = self.net(evidence.float())
        gamma = self.gamma(h).unsqueeze(1)
        beta = self.beta(h).unsqueeze(1)
        gate = torch.sigmoid(self.gate(h)).unsqueeze(1) * self.max_weight
        text = text_features.float().unsqueeze(0)
        text_mod = text + gate * (gamma * text + beta)
        return F.normalize(text_mod, dim=-1)


class RNSpatialTokenProjector(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x.float())


class CustomCLIP(nn.Module):


    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        pl_cfg = cfg.TRAINER.LSelectPROMPT
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.use_selector = bool(pl_cfg.USE_SELECTOR)
        self.selector_start_epoch = int(pl_cfg.SELECTOR_START_EPOCH)
        self.use_residual_token_logits = bool(pl_cfg.USE_RESIDUAL_TOKEN_LOGITS)
        self.use_text_film = bool(pl_cfg.USE_TEXT_FIILM)
        self.current_epoch = 0
        self.force_base_only_flag = False

        embed_dim = int(clip_model.text_projection.shape[1])
        self.embed_dim = embed_dim
        self.selector = VisualTokenSelector(cfg, embed_dim=embed_dim)
        self.token_gate = ResidualTokenLogitGate(
            dim=embed_dim,
            hidden_dim=int(pl_cfg.GATE_HIDDEN_DIM),
            dropout=float(pl_cfg.GATE_DROPOUT),
            max_weight=float(pl_cfg.TOKEN_LOGIT_WEIGHT),
            learnable=bool(pl_cfg.LEARNABLE_TOKEN_GATE),
        )
        self.text_modulator = TextFeatureFiLM(
            dim=embed_dim,
            hidden_dim=int(pl_cfg.MOD_HIDDEN_DIM),
            dropout=float(pl_cfg.MOD_DROPOUT),
            max_weight=float(pl_cfg.TEXT_FIILM_WEIGHT),
        )


        self.is_vit = hasattr(self.image_encoder, "class_embedding") and hasattr(self.image_encoder, "transformer")
        self.rn_token_proj = None
        self.rn_use_pretrained_attnpool_proj = False
        if not self.is_vit:
            if not hasattr(self.image_encoder, "attnpool"):
                raise NotImplementedError(
                    "SelectPrompt supports CLIP ViT and CLIP ModifiedResNet with attnpool. "
                    f"Got unsupported visual encoder: {type(self.image_encoder)}"
                )

            attnpool = self.image_encoder.attnpool
            required = ["v_proj", "c_proj", "positional_embedding"]
            if all(hasattr(attnpool, name) for name in required):
                self.rn_use_pretrained_attnpool_proj = True
                print("RN spatial tokens: using pretrained attnpool v_proj + c_proj")
            elif hasattr(attnpool, "c_proj"):
                # Fallback for unusual CLIP forks.
                in_dim = attnpool.c_proj.in_features
                out_dim = attnpool.c_proj.out_features
                self.rn_token_proj = RNSpatialTokenProjector(in_dim, out_dim)
                print("RN spatial tokens: using fallback trainable linear projector")
            else:
                raise NotImplementedError(
                    "RN visual encoder has attnpool, but no usable projection layers "
                    "for spatial token extraction."
                )

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
        self.selector.set_epoch(epoch)

    def set_force_base_only(self, flag: bool):
        self.force_base_only_flag = bool(flag)

    def _encode_vit_with_patch_tokens(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        visual = self.image_encoder
        x = visual.conv1(images.type(self.dtype))  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # [B, N, width]

        cls_token = visual.class_embedding.to(x.dtype)
        cls_token = cls_token + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        cls = visual.ln_post(x[:, 0, :])
        patch = visual.ln_post(x[:, 1:, :])

        if visual.proj is not None:
            cls = cls @ visual.proj
            patch = patch @ visual.proj

        return cls, patch

    def _project_rn_spatial_tokens(self, feature_map: torch.Tensor) -> torch.Tensor:

        visual = self.image_encoder
        attnpool = visual.attnpool
        bsz, channels, height, width = feature_map.shape
        tokens = feature_map.reshape(bsz, channels, height * width).permute(0, 2, 1)  # [B, HW, C]

        if self.rn_use_pretrained_attnpool_proj:
            cls_token = tokens.mean(dim=1, keepdim=True)
            seq = torch.cat([cls_token, tokens], dim=1)  # [B, 1+HW, C]

            pos = attnpool.positional_embedding.to(dtype=seq.dtype, device=seq.device)
            if pos.shape[0] == seq.shape[1]:
                seq = seq + pos.unsqueeze(0)
            else:
                # This should not happen for normal RN50 with 224x224 input.
                # Keep running instead of crashing if a custom resolution is used.
                print(
                    f"Warning: RN attnpool positional length {pos.shape[0]} "
                    f"does not match token length {seq.shape[1]}; skip positional add."
                )

            spatial = seq[:, 1:, :]
            spatial = attnpool.v_proj(spatial)
            spatial = attnpool.c_proj(spatial)
            return spatial.float()

        assert self.rn_token_proj is not None
        return self.rn_token_proj(tokens).float()

    def _encode_rn_with_spatial_tokens(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        visual = self.image_encoder
        x = images.type(self.dtype)

        def stem(z):

            if hasattr(visual, "relu1") and hasattr(visual, "relu2") and hasattr(visual, "relu3"):
                z = visual.relu1(visual.bn1(visual.conv1(z)))
                z = visual.relu2(visual.bn2(visual.conv2(z)))
                z = visual.relu3(visual.bn3(visual.conv3(z)))
            else:
                z = visual.relu(visual.bn1(visual.conv1(z)))
                z = visual.relu(visual.bn2(visual.conv2(z)))
                z = visual.relu(visual.bn3(visual.conv3(z)))

            z = visual.avgpool(z)
            return z

        x = stem(x)
        x = visual.layer1(x)
        x = visual.layer2(x)
        x = visual.layer3(x)
        x = visual.layer4(x)

        tokens = self._project_rn_spatial_tokens(x)
        global_feature = visual.attnpool(x)
        return global_feature, tokens

    def encode_image_with_tokens(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_vit:
            return self._encode_vit_with_patch_tokens(images)
        return self._encode_rn_with_spatial_tokens(images)

    def encode_text_features(self) -> torch.Tensor:
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts.to(prompts.device)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        return text_features

    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = "eval",
        return_aux: bool = False,
        force_base_only: bool = False,
    ):
        image_features, patch_tokens = self.encode_image_with_tokens(images)
        text_features = self.encode_text_features()

        image_features = F.normalize(image_features.float(), dim=-1)
        patch_tokens = patch_tokens.float()
        text_features = F.normalize(text_features.float(), dim=-1)
        logit_scale = self.logit_scale.exp().float()

        base_logits = logit_scale * image_features @ text_features.t()

        aux = {
            "selected_ratio": torch.tensor(1.0, device=images.device),
            "budget_loss": torch.tensor(0.0, device=images.device),
            "selection_entropy": torch.tensor(0.0, device=images.device),
            "token_gate": torch.tensor(0.0, device=images.device),
            "selector_on": torch.tensor(0.0, device=images.device),
        }

        selector_on = (
            self.use_selector
            and (not force_base_only)
            and (not self.force_base_only_flag)
            and (self.current_epoch >= self.selector_start_epoch)
        )

        if not selector_on:
            return (base_logits, aux) if return_aux else base_logits

        evidence, _, sel_aux = self.selector(
            patch_tokens=patch_tokens,
            text_features=text_features,
            base_logits=base_logits,
            labels=labels,
            mode=mode,
        )
        evidence = F.normalize(evidence.float(), dim=-1)

        logits = base_logits
        token_gate = torch.tensor(0.0, device=images.device)

        if self.use_residual_token_logits:
            token_logits = logit_scale * evidence @ text_features.t()
            token_gate = self.token_gate(image_features, evidence)  # [B, 1]

            logits =logits + lambda*token_gate * token_logits
        if self.use_text_film:
            text_features_x = self.text_modulator(text_features, evidence)
            film_logits = logit_scale * torch.einsum("bd,bcd->bc", image_features, text_features_x)
            logits = logits + film_logits

        aux.update(sel_aux)
        aux["token_gate"] = token_gate.detach().mean()
        aux["selector_on"] = torch.tensor(1.0, device=images.device)
        return (logits, aux) if return_aux else logits


@TRAINER_REGISTRY.register()
class SelectPrompt(TrainerX):


    def __init__(self, cfg):
        super().__init__(cfg)
        self.gce_loss = GeneralizedCrossEntropy(q=1.0)
        self.GCE_loss = self.gce_loss
        self.num_equal = []
        self.confident_rate = []
        self.clean_rate = []
        self.best_acc = -1
        self.best_epoch = -1
        self.test_acc = []

    def check_cfg(self, cfg):
        assert cfg.TRAINER.SelectPROMPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP backbone: {cfg.MODEL.BACKBONE.NAME}")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.SelectPROMPT.PREC in ["fp32", "amp"]:
            clip_model.float()


        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Freezing CLIP encoders; training prompt learner + selector/gate/modulator only")
        trainable_keywords = [
            "prompt_learner",
            "selector",
            "token_gate",
            "text_modulator",
            "rn_token_proj",
        ]
        for name, param in self.model.named_parameters():
            param.requires_grad_(any(key in name for key in trainable_keywords))

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / total_params:.4f}%)")

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("selectprompt", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.SelectPROMPT.PREC == "amp" else None

    def parse_batch_train(self, batch):
        images = batch["img"].to(self.device)
        labels = batch["label"].to(self.device)
        gt_labels = batch["gttarget"].to(self.device)
        return images, labels, gt_labels

    def _extra_light_losses(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg.TRAINER.SelectPROMPT
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        loss = loss + float(cfg.BUDGET_LOSS_WEIGHT) * aux.get("budget_loss", 0.0)
        # Small negative entropy term: maximize selector entropy a little, avoids early collapse.
        loss = loss - float(cfg.SELECTION_ENTROPY_WEIGHT) * aux.get("selection_entropy", 0.0)
        return loss

    def _forward_backward(self, batch, loss_fn, loss_key, acc_key, mode: str):
        images, labels, _ = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.SelectPROMPT.PREC
        self.model.set_epoch(self.epoch)

        if prec == "amp":
            with autocast():
                logits, aux = self.model(images, labels=labels, mode=mode, return_aux=True)
                loss_cls = loss_fn(logits, labels)
                loss = loss_cls + self._extra_light_losses(aux)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits, aux = self.model(images, labels=labels, mode=mode, return_aux=True)
            loss_cls = loss_fn(logits, labels)
            loss = loss_cls + self._extra_light_losses(aux)
            self.model_backward_and_update(loss)

        return {
            loss_key: float(loss.detach().item()),
            f"{loss_key}_cls": float(loss_cls.detach().item()),
            acc_key: compute_accuracy(logits, labels)[0].item(),
            "sel_ratio": float(aux["selected_ratio"].detach().item()),
            "sel_entropy": float(aux["selection_entropy"].detach().item()),
            "token_gate": float(aux["token_gate"].detach().item()),
            "selector_on": float(aux["selector_on"].detach().item()),
        }

    def forward_backward_ce(self, batch):
        return self._forward_backward(
            batch=batch,
            loss_fn=F.cross_entropy,
            loss_key="loss_x",
            acc_key="acc_x",
            mode="clean",
        )

    def forward_backward_mae(self, batch):
        return self._forward_backward(
            batch=batch,
            loss_fn=self.gce_loss,
            loss_key="loss_u",
            acc_key="acc_u",
            mode="noisy",
        )

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            try:
                checkpoint = load_checkpoint(model_path)
            except Exception as exc:
                print(f"Dassl load_checkpoint failed with {type(exc).__name__}: {exc}")
                print("Fallback to torch.load(..., weights_only=False)")
                checkpoint = _torch_load_safe(model_path, map_location="cpu")

            state_dict = checkpoint["state_dict"]
            load_epoch = checkpoint.get("epoch", -1)

            # Ignore fixed token vectors; they depend on current class names.
            for key in list(state_dict.keys()):
                if key.endswith("token_prefix") or key.endswith("token_suffix"):
                    del state_dict[key]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {load_epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)

    @staticmethod
    def _build_train_loader_iter(loader):
        if loader is None:
            return None, 0
        return iter(loader), len(loader)

    def _log_batch_progress(self, losses, batch_time, data_time, num_batches, loss_label):
        eta_seconds = batch_time.avg * (num_batches - self.batch_idx - 1)
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        info = [
            f"epoch [{self.epoch + 1}/{self.max_epoch}]",
            f"batch [{self.batch_idx + 1}/{num_batches}]",
            f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
            f"data {data_time.val:.3f} ({data_time.avg:.3f})",
            f"{loss_label} {losses}",
            f"lr {self.get_current_lr():.4e}",
            f"eta {eta}",
        ]
        print(" ".join(info))

    def _run_train_loader(
        self,
        loader_iter,
        num_batches,
        forward_fn,
        losses,
        batch_time,
        data_time,
        scalar_prefix,
        loss_label,
        end_time,
    ):
        for self.batch_idx in range(num_batches):
            try:
                batch = next(loader_iter)
            except StopIteration:
                break

            data_time.update(time.time() - end_time)
            loss_summary = forward_fn(batch)
            losses.update(loss_summary)
            batch_time.update(time.time() - end_time)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or num_batches < self.cfg.TRAIN.PRINT_FREQ:
                self._log_batch_progress(losses, batch_time, data_time, num_batches, loss_label)

            n_iter = self.epoch * max(self.num_batches_x + self.num_batches_u, 1) + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar(f"{scalar_prefix}/{name}", meter.avg, n_iter)

            end_time = time.time()
        return end_time

    @staticmethod
    def _delete_indices(data_source, indices):
        for index in sorted(indices, reverse=True):
            del data_source[index]

    def _compute_ot_budget(self, curriclum_epoch, begin_rate, curriclum_mode):
        if self.epoch < curriclum_epoch:
            budget, _ = curriculum_scheduler(
                self.epoch,
                curriclum_epoch,
                begin=begin_rate,
                end=1,
                mode=curriclum_mode,
            )
        else:
            budget = 1.0
        return budget

    def _get_ot_cfg(self):
        cfg = self.cfg.DATASET
        return {
            "reg_feat": cfg.REG_FEAT,
            "reg_lab": cfg.REG_LAB,
            "curriclum_epoch": cfg.CURRICLUM_EPOCH,
            "begin_rate": cfg.BEGIN_RATE,
            "curriclum_mode": cfg.CURRICLUM_MODE,
            "pmode": cfg.PMODE,
            "reg_e": getattr(cfg, "REG_E", 0.001),
        }

    @staticmethod
    def _print_ot_overview(gt_labels, argmax_plabels):
        print("before epoch:data num:", len(gt_labels))
        print(
            "before epoch:different number:",
            np.sum(gt_labels.cpu().numpy() != argmax_plabels.cpu().numpy()),
        )

    @staticmethod
    def _build_unlabeled_mask(argmax_plabels, noisy_labels, selected_mask):
        conf_l_mask, conf_u_mask, lowconf_u_mask = get_masks(argmax_plabels, noisy_labels, None, selected_mask)
        selected_rate_conf_l, _, _ = output_selected_rate(conf_l_mask, conf_u_mask, lowconf_u_mask)
        print("confident_label rate", selected_rate_conf_l)
        unlabeled_mask = torch.logical_or(conf_u_mask, lowconf_u_mask)
        return conf_l_mask, unlabeled_mask

    def _run_ot_pseudolabeling(self, budget, reg_feat, reg_lab, pmode, reg_e):
        self.model.set_epoch(self.epoch)
        # Use the stable base prompt logits for PromptOT splitting. Otherwise a weak
        # early selector can corrupt clean/noisy partitioning.
        self.model.set_force_base_only(True)
        try:
            with torch.no_grad():
                _, noisy_labels, gt_labels, selected_mask, _, argmax_plabels = OT_PL(
                    self.model,
                    self.train_loader_x,
                    num_class=self.cfg.DATASET.num_class,
                    batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                    budget=budget,
                    reg_feat=reg_feat,
                    reg_lab=reg_lab,
                    Pmode=pmode,
                    reg_e=reg_e,
                    load_all=True,
                )
        finally:
            self.model.set_force_base_only(False)
        return noisy_labels, gt_labels, selected_mask, argmax_plabels

    def _apply_ot_split(self, conf_l_mask, unlabeled_mask, pred_labels, gt_labels):
        conf_mask_np = conf_l_mask.cpu().numpy()
        unlabeled_mask_np = unlabeled_mask.cpu().numpy()
        pred_np = pred_labels.cpu().numpy()
        gt_np = gt_labels.cpu().numpy()

        clean_total = int(conf_mask_np.sum())
        clean_correct = int(np.sum((pred_np == gt_np) & conf_mask_np))
        noisy_total = int(unlabeled_mask_np.sum())
        noisy_correct = int(np.sum((pred_np == gt_np) & unlabeled_mask_np))
        clean_rate = clean_correct / max(clean_total, 1)
        self.clean_rate.append(clean_rate)

        confident_indices = np.nonzero(conf_mask_np)[0]
        unlabeled_indices = np.nonzero(unlabeled_mask_np)[0]

        self.tmp_train_loader_x = copy.deepcopy(self.train_loader_x)
        self.train_loader_u = copy.deepcopy(self.train_loader_x)

        print("before: len(self.train)", len(self.train_loader_x.dataset.data_source))
        print("before: len of confident samples", len(confident_indices))
        print(f"clean_rate:{clean_rate}")
        print(f"noisy_correct:{noisy_correct}/{noisy_total}")

        self._delete_indices(self.train_loader_x.dataset.data_source, unlabeled_indices)
        print("after delete: len(clean_dataset)", len(self.train_loader_x.dataset.data_source))
        self._delete_indices(self.train_loader_u.dataset.data_source, confident_indices)
        print("after delete: len(noisy_dataset)", len(self.train_loader_u.dataset.data_source))

    def _restore_train_loaders(self):
        self.train_loader_x = copy.deepcopy(self.tmp_train_loader_x)
        self.train_loader_u = copy.deepcopy(self.tmp_train_loader_x)
        print("after epoch: len(clean dataset)", len(self.train_loader_x.dataset.data_source))
        print("after epoch: len(noisy dataset)", len(self.train_loader_u.dataset.data_source))

    def before_epoch(self):
        self.model.set_epoch(self.epoch)
        if not self.cfg.DATASET.USE_OT:
            self.train_loader_u = None
            return

        epoch_start = time.time()
        ot_cfg = self._get_ot_cfg()
        budget = self._compute_ot_budget(
            ot_cfg["curriclum_epoch"],
            ot_cfg["begin_rate"],
            ot_cfg["curriclum_mode"],
        )

        ot_start = time.time()
        noisy_labels, gt_labels, selected_mask, argmax_plabels = self._run_ot_pseudolabeling(
            budget=budget,
            reg_feat=ot_cfg["reg_feat"],
            reg_lab=ot_cfg["reg_lab"],
            pmode=ot_cfg["pmode"],
            reg_e=ot_cfg["reg_e"],
        )
        print(f"before epoch: OT_PL time {time.time() - ot_start:.2f}s")
        self._print_ot_overview(gt_labels, argmax_plabels)

        conf_l_mask, unlabeled_mask = self._build_unlabeled_mask(argmax_plabels, noisy_labels, selected_mask)
        if conf_l_mask.sum().item() > 0:
            self._apply_ot_split(conf_l_mask, unlabeled_mask, argmax_plabels, gt_labels)
        else:
            self.train_loader_u = None
            print("No confident clean samples selected by OT this epoch; train on original train_loader_x only.")

        print(f"before epoch total time {time.time() - epoch_start:.2f}s")

    def run_epoch(self):
        self.model.set_epoch(self.epoch)
        self.set_model_mode("train")

        losses_x = MetricMeter()
        losses_u = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        train_loader_x_iter, len_train_loader_x = self._build_train_loader_iter(self.train_loader_x)
        train_loader_u_iter, len_train_loader_u = self._build_train_loader_iter(getattr(self, "train_loader_u", None))

        self.num_batches_x = len_train_loader_x
        self.num_batches_u = len_train_loader_u
        end = time.time()

        if train_loader_x_iter is not None:
            end = self._run_train_loader(
                loader_iter=train_loader_x_iter,
                num_batches=self.num_batches_x,
                forward_fn=self.forward_backward_ce,
                losses=losses_x,
                batch_time=batch_time,
                data_time=data_time,
                scalar_prefix="train_x",
                loss_label="loss_x",
                end_time=end,
            )

        if train_loader_u_iter is not None:
            self._run_train_loader(
                loader_iter=train_loader_u_iter,
                num_batches=self.num_batches_u,
                forward_fn=self.forward_backward_mae,
                losses=losses_u,
                batch_time=batch_time,
                data_time=data_time,
                scalar_prefix="train_u",
                loss_label="loss_u",
                end_time=end,
            )

        self.update_lr()

    def after_epoch(self):
        epoch_end_start = time.time()
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0
            else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            self.model.set_epoch(self.epoch)
            test_start = time.time()
            curr_result = self.test(split="val")
            print(f"after epoch: test time {time.time() - test_start:.2f}s")

            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar",
                )

        if meet_checkpoint_freq or last_epoch:
            save_start = time.time()
            self.save_model(self.epoch, self.output_dir)
            print(f"after epoch: save time {time.time() - save_start:.2f}s")

        if self.cfg.DATASET.USE_OT and hasattr(self, "tmp_train_loader_x"):
            self._restore_train_loaders()

        print(f"after epoch total time {time.time() - epoch_end_start:.2f}s")