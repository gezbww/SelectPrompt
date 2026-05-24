import argparse

import torch
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import collect_env_info, set_random_seed, setup_logger


import datasets.oxford_pets
import datasets.oxford_flowers

import datasets.dtd
import datasets.eurosat

import datasets.food101

import datasets.caltech101

import datasets.food101n



import trainers.selectprompt


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print(f"{key}: {args.__dict__[key]}")
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.resume:
        cfg.RESUME = args.resume
    if args.seed is not None:
        cfg.SEED = args.seed
    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains
    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """Add SelectPrompt config keys before merging yaml files."""
    from yacs.config import CfgNode as CN


    cfg.TRAINER.SelectPrompt = CN()
    cfg.TRAINER.SelectPrompt.N_CTX = 16
    cfg.TRAINER.SelectPrompt.CSC = False
    cfg.TRAINER.SelectPrompt.CTX_INIT = ""
    cfg.TRAINER.SelectPrompt.PREC = "fp16"
    cfg.TRAINER.SelectPrompt.CLASS_TOKEN_POSITION = "end"

    # New trainer keys.
    cfg.TRAINER.SelectPrompt = CN()
    cfg.TRAINER.SelectPrompt.N_CTX = 16
    cfg.TRAINER.SelectPrompt.CSC = False
    cfg.TRAINER.SelectPrompt.CTX_INIT = ""
    cfg.TRAINER.SelectPrompt.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.SelectPrompt.CLASS_TOKEN_POSITION = "end"

    # LightVLA-style token selection.
    cfg.TRAINER.SelectPrompt.USE_SELECTOR = True
    cfg.TRAINER.SelectPrompt.SELECTOR_START_EPOCH = 10
    cfg.TRAINER.SelectPrompt.KEEP_RATIO = 0.45
    cfg.TRAINER.SelectPrompt.MIN_KEEP_TOKENS = 32
    cfg.TRAINER.SelectPrompt.MAX_KEEP_TOKENS = 96
    cfg.TRAINER.SelectPrompt.TOPR_QUERY_CLASSES = 3
    cfg.TRAINER.SelectPrompt.GUMBEL_TEMP = 1.0
    cfg.TRAINER.SelectPrompt.GUMBEL_MIN_TEMP = 0.3

    # Stable residual enhancement branch.
    cfg.TRAINER.SelectPrompt.USE_RESIDUAL_TOKEN_LOGITS = True
    cfg.TRAINER.SelectPrompt.TOKEN_LOGIT_WEIGHT = 0.30
    cfg.TRAINER.SelectPrompt.LEARNABLE_TOKEN_GATE = True
    cfg.TRAINER.SelectPrompt.GATE_HIDDEN_DIM = 256
    cfg.TRAINER.SelectPrompt.GATE_DROPOUT = 0.10

    # Optional text FiLM branch. Off by default for stability.
    cfg.TRAINER.SelectPrompt.USE_TEXT_FIILM = False
    cfg.TRAINER.SelectPrompt.TEXT_FIILM_WEIGHT = 0.10
    cfg.TRAINER.SelectPrompt.MOD_HIDDEN_DIM = 512
    cfg.TRAINER.SelectPrompt.MOD_DROPOUT = 0.10

    # Loss weights.
    cfg.TRAINER.SelectPrompt.BUDGET_LOSS_WEIGHT = 0.005
    cfg.TRAINER.SelectPrompt.SELECTION_ENTROPY_WEIGHT = 0.0005
    cfg.TRAINER.SelectPrompt.TEXT_REG_WEIGHT = 0.0
    cfg.TRAINER.SelectPrompt.LOG_TOKEN_STATS = True

    # Dataset/noise/OT defaults copied from SelectPrompt.
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.NUM_SHOTS = 16
    cfg.DATASET.NOISE_LABEL = True
    cfg.DATASET.NOISE_RATE = 0.5
    cfg.DATASET.NOISE_TYPE = "sym"
    cfg.DATASET.num_class = 100

    cfg.DATASET.USE_OT = True
    cfg.DATASET.REG_FEAT = 1.0
    cfg.DATASET.REG_LAB = 1.0
    cfg.DATASET.CURRICLUM_EPOCH = 0
    cfg.DATASET.BEGIN_RATE = 0.3
    cfg.DATASET.CURRICLUM_MODE = "linear"
    cfg.DATASET.PMODE = "logP"
    cfg.DATASET.REG_E = 0.001


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        print(f"Setting fixed seed: {cfg.SEED}")
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/DATA/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="output/eurosat/SelectPrompt", help="output directory")
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory")
    parser.add_argument("--seed", type=int, default=1, help="positive value enables a fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+")
    parser.add_argument("--target-domains", type=str, nargs="+")
    parser.add_argument("--transforms", type=str, nargs="+")
    parser.add_argument("--config-file", type=str, default="configs/trainers/SelectPrompt/vit_b16.yaml")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/eurosat.yaml")
    parser.add_argument("--trainer", type=str, default="SelectPrompt")
    parser.add_argument("--backbone", type=str, default="")
    parser.add_argument("--head", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--load-epoch", type=int)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    main(parser.parse_args())
