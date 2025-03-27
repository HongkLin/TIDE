# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_tide_config(cfg):
    #TIDE
    cfg.TIDE = CN()
    cfg.TIDE.SEED = 42
    cfg.TIDE.PRETRAINED_DIFFUSION_MODEL_WEIGHT = "pretrained_model/PixArt-XL-2-512x512"

    #instruction
    cfg.TIDE.INSTRUCT = CN()
    cfg.TIDE.NUM_IMAGE_PER_PROMPT = 1
    cfg.TIDE.REGION_FILTER_TH = 100
    cfg.TIDE.DISTANCE_MAP_TH = None
    cfg.TIDE.RETURN_CRF_REFINE = True
    cfg.TIDE.TEMPERATURE = 40
