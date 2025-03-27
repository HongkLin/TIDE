# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Fine-tuning script for Stable Diffusion for text2image with HuggingFace diffusers."""
import argparse
import logging
import math
import copy
import gc
import os
import sys
import random
import shutil
from pathlib import Path
from typing import List, Union

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo
from PIL import Image
from packaging import version
from peft import LoraConfig, get_peft_model_state_dict, get_peft_model, PeftModel
from torchvision import transforms
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

script_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_path = os.path.join(script_dir, 'pipeline')
sys.path.insert(0, subfolder_path)

from utils import IDColour
from pipeline.tide_transformer import (
    PixArtSpecialAttnTransformerModel,
    MiniTransformerModel,
    TIDETransformerModel,
    TIDE_TANs,
)

from pipeline.pipeline_tide import TIDEPipeline

cmap = plt.get_cmap('Spectral_r')

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.29.2")

logger = get_logger(__name__, log_level="INFO")


def log_validation(vae, tide_transformer, tokenizer, scheduler, text_encoder, args, accelerator, weight_dtype,
                   step, is_final_validation=False):

    logger.info(f"Running validation step {step} ... ")

    unwrap_tide_transformer = accelerator.unwrap_model(tide_transformer, keep_fp32_wrapper=False)

    pipeline = TIDEPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        transformer=unwrap_tide_transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []

    for validation_text_prompt in args.validation_prompts:

        target_images = []
        depth_images = []
        mask_images = []

        for _ in range(args.num_validation_images):
            target_image, depth_image, mask_image = pipeline(
                prompt=validation_text_prompt,
                num_inference_steps=20,
                generator=generator,
                guidance_scale=3.0,
                zero_output_type='np'
            )
            target_images.append(target_image.images[0])
            depth_images.append(
                Image.fromarray(
                    np.uint8(
                        255 * cmap(np.mean(depth_image.images[0], axis=-1))
                    )
                )
            )
            mask_images.append(mask_image.images[0])

        image_logs.append(
            {
                "text_prompt": validation_text_prompt,
                "target_images": target_images,
                "depth_images": depth_images,
                "mask_images": mask_images,
            }
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                validation_prompt = log["text_prompt"]
                target_images = log["target_images"]
                depth_images = log["depth_images"]
                mask_images = log["mask_images"]

                for target_image, depth_image, mask_image in zip(target_images, depth_images, mask_images):
                    target_image = wandb.Image(target_image, caption=f"image-" + validation_prompt)
                    depth_image = wandb.Image(depth_image, caption=f"depth-" + validation_prompt)
                    mask_image = wandb.Image(mask_image, caption=f"mask-" + validation_prompt)

                    formatted_images.append(target_image)
                    formatted_images.append(depth_image)
                    formatted_images.append(mask_image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Validation done!!")

        return image_logs


def pyramid_noise_like(x, discount=0.9):
    b, c, w, h = x.shape  # EDIT: w and h get over-written, rename for a different variant!
    u = torch.nn.Upsample(size=(w, h), mode='bilinear')
    noise = torch.randn_like(x)
    for i in range(10):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        w, h = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))
        noise += u(torch.randn(b, c, w, h).to(x)) * discount ** i
        if w == 1 or h == 1: break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # parser = deepspeed.add_config_arguments(parser)
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='./pretrained_model/PixArt-XL-2-512x512',
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--pyramid_noise",
        action="store_true",
        help="Whether or not to use pyramid_noise.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default='./datasets/UIEB_triplets',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the place conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=['a coral reef with many fish swimming around it.', 'a stingray on a sandy beach.'],
        nargs="+",
        help=(
            "A set of paths to the place conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_image`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_image` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/train_tide",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--max_id_num",
        type=int,
        default=8,
        help=(
            "The max num semantic categories"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help=(
            'wandb name'
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="tide",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    args = parser.parse_args()

    # Sanity checks
    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args

def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True,
                                  token=args.hub_token).repo_id

    # See Section 3.1. of the paper.
    max_length = 120

    # For mixed precision training we cast all non-trainable weigths (vae, text_encoder) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",
                                                    torch_dtype=weight_dtype)
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer",
                                            revision=args.revision, torch_dtype=weight_dtype)

    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder",
                                                  revision=args.revision, torch_dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,
                                        variant=args.variant, torch_dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.to(accelerator.device)

    transformer = PixArtSpecialAttnTransformerModel.from_pretrained(args.pretrained_model_name_or_path,
                                                                    subfolder="transformer", torch_dtype=weight_dtype)
    transformer.requires_grad_(False)

    depth_transformer = MiniTransformerModel.from_config(
        "pretrained_model/TIDE_MiniTransformer",
        subfolder="mini_transformer",
        torch_dtype=weight_dtype
    )
    depth_transformer = depth_transformer.to(weight_dtype)
    _state_dict = torch.load(
        os.path.join("pretrained_model/TIDE_MiniTransformer/mini_transformer", 'diffusion_pytorch_model.pth'),
        map_location='cpu'
    )
    depth_transformer.load_state_dict(_state_dict)
    depth_transformer.requires_grad_(False)
    del _state_dict

    mask_transformer = copy.deepcopy(depth_transformer)

    tan_modules = TIDE_TANs(num_layers=10, time_adaptive=True)
    tan_modules.to(accelerator.device)
    tan_modules.train()
    # Freeze the transformer parameters before adding adapters
    for param in transformer.parameters():
        param.requires_grad_(False)

    image_lora_config = LoraConfig(
        r=32,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "proj",
            "linear",
            "linear_1",
            "linear_2",
        ],
    )
    depth_lora_config = LoraConfig(
        r=64,
        init_lora_weights="gaussian",
        target_modules=[
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn2.to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "proj",
            "linear",
            "linear_1",
            "linear_2",
        ],
    )
    mask_lora_config = LoraConfig(
        r=64,
        init_lora_weights="gaussian",
        target_modules=[
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn2.to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "proj",
            "linear",
            "linear_1",
            "linear_2",
        ],
    )


    transformer.to(accelerator.device)
    depth_transformer.to(accelerator.device)
    mask_transformer.to(accelerator.device)
    def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
        if not isinstance(model, list):
            model = [model]
        for m in model:
            for param in m.parameters():
                # only upcast trainable parameters into fp32
                if param.requires_grad:
                    param.data = param.to(dtype)

    transformer_image = get_peft_model(transformer, image_lora_config)
    transformer_depth = get_peft_model(depth_transformer, depth_lora_config)
    transformer_mask = get_peft_model(mask_transformer, mask_lora_config)

    # transformer_l2i = transformer
    if accelerator.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(transformer_image, dtype=torch.float32)
        cast_training_params(transformer_depth, dtype=torch.float32)
        cast_training_params(transformer_mask, dtype=torch.float32)

    transformer_image.print_trainable_parameters()
    transformer_depth.print_trainable_parameters()
    transformer_mask.print_trainable_parameters()
    transformer_image.train()
    transformer_depth.train()
    transformer_mask.train()
    #

    tide_transformer = TIDETransformerModel(transformer_image, transformer_depth, transformer_mask, tan_modules, training=True)
    params_to_optimize = filter(lambda p: p.requires_grad, tide_transformer.parameters())

    def unwrap_model(model, keep_fp32_wrapper=True):
        model = accelerator.unwrap_model(model, keep_fp32_wrapper=keep_fp32_wrapper)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                tide_transformer_ = accelerator.unwrap_model(tide_transformer)
                transformer_image_ = tide_transformer_.transformer_image
                transformer_depth_ = tide_transformer_.transformer_depth
                transformer_mask_ = tide_transformer_.transformer_mask
                tan_modules_ = tide_transformer_.tan_modules

                image_lora_state_dict = get_peft_model_state_dict(transformer_image_, adapter_name="default")
                StableDiffusionPipeline.save_lora_weights(os.path.join(output_dir, "image_transformer_lora"),
                                                          image_lora_state_dict)
                depth_lora_state_dict = get_peft_model_state_dict(transformer_depth_, adapter_name="default")
                StableDiffusionPipeline.save_lora_weights(os.path.join(output_dir, "depth_transformer_lora"),
                                                          depth_lora_state_dict)
                mask_lora_state_dict = get_peft_model_state_dict(transformer_mask_, adapter_name="default")
                StableDiffusionPipeline.save_lora_weights(os.path.join(output_dir, "mask_transformer_lora"),
                                                          mask_lora_state_dict)

                transformer_image_.save_pretrained(os.path.join(output_dir, "image_transformer_lora"))
                transformer_depth_.save_pretrained(os.path.join(output_dir, "depth_transformer_lora"))
                transformer_mask_.save_pretrained(os.path.join(output_dir, "mask_transformer_lora"))
                tan_modules_.save_pretrained(os.path.join(output_dir, "tan_modules"))

        def load_model_hook(models, input_dir):
            # load the LoRA into the model
            transformer_ = accelerator.unwrap_model(transformer)
            transformer_.load_adapter(input_dir, "default", is_trainable=True)

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        transformer_image.enable_gradient_checkpointing()
        transformer_depth.enable_gradient_checkpointing()
        transformer_mask.enable_gradient_checkpointing()
        tan_modules.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset = load_dataset(
        path="./datasets/tide_uwdense.py",
        trust_remote_code=True
    )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True, proportion_empty_prompts=0., max_length=120):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])

        inputs = tokenizer(captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids, inputs.attention_mask

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    depth_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(args.resolution),
            transforms.PILToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    semantic_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(args.resolution),
            IDColour(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def selfnorm(arr):
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        return Image.fromarray(arr)

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]

        depth_images = [np.load(path) for path in examples['depth_image']]
        depth_images = [selfnorm(depth) for depth in depth_images]
        depth_images = [depth_transforms(depth) for depth in depth_images]
        examples["depth_pixel_values"] = [depth.repeat(3, 1, 1) for depth in depth_images]

        mask_images = [mask for mask in examples['semantic_image']]
        examples["mask_pixel_values"] = [semantic_image_transforms(mask) for mask in mask_images]


        examples["input_ids"], examples['prompt_attention_mask'] = tokenize_captions(examples,
                                                                                     proportion_empty_prompts=args.proportion_empty_prompts,
                                                                                     max_length=max_length)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        depth_pixel_values = torch.stack([example["depth_pixel_values"] for example in examples])
        depth_pixel_values = depth_pixel_values.to(memory_format=torch.contiguous_format).float()

        mask_pixel_values = torch.stack([example["mask_pixel_values"] for example in examples])
        mask_pixel_values = mask_pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.stack([example["input_ids"] for example in examples])
        prompt_attention_mask = torch.stack([example["prompt_attention_mask"] for example in examples])

        return {
            "pixel_values": pixel_values,
            "depth_pixel_values": depth_pixel_values,
            "mask_pixel_values": mask_pixel_values,
            "input_ids": input_ids,
            'prompt_attention_mask': prompt_attention_mask
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    tide_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(tide_transformer, optimizer,
                                                                                      train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.tracker_project_name,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_name}}
        )

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    latent_channels = transformer.config.in_channels
    for epoch in range(first_epoch, args.num_train_epochs):
        tide_transformer.train()
        train_total_loss = 0.0
        train_image_loss = 0.0
        train_depth_loss = 0.0
        train_mask_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(tide_transformer):
                # Convert label images to latent space
                depth_latents = vae.encode(batch["depth_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                depth_latents = depth_latents * vae.config.scaling_factor

                mask_latents = vae.encode(batch["mask_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                mask_latents = mask_latents * vae.config.scaling_factor

                # Convert images to latent space
                image_latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                image_latents = image_latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                if args.pyramid_noise:
                    depth_noise = pyramid_noise_like(depth_latents)
                    mask_noise = pyramid_noise_like(mask_latents)
                    image_noise = pyramid_noise_like(image_latents)
                else:
                    depth_noise = torch.randn_like(depth_latents)
                    mask_noise = torch.randn_like(mask_latents)
                    image_noise = torch.randn_like(image_latents)

                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    depth_noise += args.noise_offset * torch.randn(
                        (depth_latents.shape[0], depth_latents.shape[1], 1, 1), device=depth_latents.device
                    )
                    mask_noise += args.noise_offset * torch.randn(
                        (mask_latents.shape[0], mask_latents.shape[1], 1, 1), device=mask_latents.device
                    )
                    image_noise += args.noise_offset * torch.randn(
                        (image_latents.shape[0], image_latents.shape[1], 1, 1), device=image_latents.device
                    )

                bsz = image_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                          device=image_latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                noisy_depth_latents = noise_scheduler.add_noise(depth_latents, depth_noise, timesteps)
                noisy_mask_latents = noise_scheduler.add_noise(mask_latents, mask_noise, timesteps)
                noisy_image_latents = noise_scheduler.add_noise(image_latents, image_noise, timesteps)

                # Get the text embedding for conditioning
                prompt_embeds = text_encoder(batch["input_ids"], attention_mask=batch['prompt_attention_mask'])[0]
                prompt_attention_mask = batch['prompt_attention_mask']
                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    depth_target = depth_noise
                    mask_target = mask_noise
                    image_target = image_noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    depth_target = noise_scheduler.get_velocity(depth_latents, depth_noise, timesteps)
                    mask_target = noise_scheduler.get_velocity(mask_latents, mask_noise, timesteps)
                    image_target = noise_scheduler.get_velocity(image_latents, image_noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Prepare micro-conditions.
                added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
                if getattr(transformer, 'module', transformer).config.sample_size == 128:
                    resolution = torch.tensor([args.resolution, args.resolution]).repeat(bsz, 1)
                    aspect_ratio = torch.tensor([float(args.resolution / args.resolution)]).repeat(bsz, 1)
                    resolution = resolution.to(dtype=weight_dtype, device=depth_latents.device)
                    aspect_ratio = aspect_ratio.to(dtype=weight_dtype, device=depth_latents.device)
                    added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

                # Predict the noise residual and compute loss
                image_noise_output, depth_noise_output, mask_noise_output = tide_transformer(
                    noisy_image_latents,
                    noisy_depth_latents,
                    noisy_mask_latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timesteps,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )

                image_noise_output = image_noise_output.chunk(2, dim=1)[0]
                depth_noise_output = depth_noise_output.chunk(2, dim=1)[0]
                mask_noise_output = mask_noise_output.chunk(2, dim=1)[0]

                image_loss = F.mse_loss(image_noise_output.float(), image_target.float(), reduction="mean")
                depth_loss = F.mse_loss(depth_noise_output.float(), depth_target.float(), reduction="mean")
                mask_loss = F.mse_loss(mask_noise_output.float(), mask_target.float(), reduction="mean")
                #TODO mask loss
                loss = image_loss + depth_loss + mask_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_image_loss = accelerator.gather(image_loss.repeat(args.train_batch_size)).mean()
                train_image_loss += avg_image_loss.item() / args.gradient_accumulation_steps

                avg_depth_loss = accelerator.gather(depth_loss.repeat(args.train_batch_size)).mean()
                train_depth_loss += avg_depth_loss.item() / args.gradient_accumulation_steps

                avg_mask_loss = accelerator.gather(mask_loss.repeat(args.train_batch_size)).mean()
                train_mask_loss += avg_mask_loss.item() / args.gradient_accumulation_steps

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_total_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = params_to_optimize
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {"total_loss": train_total_loss,
                     "image_loss": train_image_loss,
                     "layout_loss": train_depth_loss,
                     "mask_loss": train_mask_loss,
                     },
                    step=global_step)
                train_total_loss = 0.0
                train_image_loss = 0.0
                train_depth_loss = 0.0
                train_mask_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                        log_validation(vae, tide_transformer, tokenizer, noise_scheduler, text_encoder, args,
                                       accelerator, weight_dtype, global_step, is_final_validation=False)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tide_transformer = accelerator.unwrap_model(tide_transformer, keep_fp32_wrapper=False)

        transformer_image = tide_transformer.transformer_image
        transformer_depth = tide_transformer.transformer_depth
        transformer_mask = tide_transformer.transformer_mask
        tan_modules = tide_transformer.tan_modules

        transformer_image.save_pretrained(os.path.join(args.output_dir, "image_transformer_lora"))
        transformer_depth.save_pretrained(os.path.join(args.output_dir, "depth_transformer_lora"))
        transformer_mask.save_pretrained(os.path.join(args.output_dir, "mask_transformer_lora"))

        image_lora_state_dict = get_peft_model_state_dict(transformer_image)
        StableDiffusionPipeline.save_lora_weights(os.path.join(args.output_dir, "image_transformer_lora"),
                                                  image_lora_state_dict)
        depth_lora_state_dict = get_peft_model_state_dict(transformer_depth)
        StableDiffusionPipeline.save_lora_weights(os.path.join(args.output_dir, "depth_transformer_lora"),
                                                  depth_lora_state_dict)
        mask_lora_state_dict = get_peft_model_state_dict(transformer_mask)
        StableDiffusionPipeline.save_lora_weights(os.path.join(args.output_dir, "mask_transformer_lora"),
                                                  mask_lora_state_dict)

        tan_modules.save_pretrained(os.path.join(args.output_dir, "tan_modules"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
