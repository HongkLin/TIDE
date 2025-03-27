import numpy as np
import torch
from PIL import Image
import os
import copy
import argparse
from matplotlib import pyplot as plt

from peft import PeftModel

from tide.utils.dataset_palette import USIS10K_COLORS_PALETTE
from tide.utils import mask_postprocess

from tide.pipeline.tide_transformer import (
    PixArtSpecialAttnTransformerModel,
    TIDETransformerModel,
    TIDE_TANs,
    MiniTransformerModel
)
from tide.pipeline.pipeline_tide import TIDEPipeline

cmap = plt.get_cmap('Spectral_r')


def colour_and_vis_id_map(id_map, palette, out_path):
    id_map = id_map.astype(np.uint8)
    id_map -= 1
    ids = np.unique(id_map)
    valid_ids = np.delete(ids, np.where(ids == 255))

    colour_layout = np.zeros((id_map.shape[0], id_map.shape[1], 3), dtype=np.uint8)
    for id in valid_ids:
        colour_layout[id_map == id, :] = palette[id].reshape(1, 3)
    colour_layout = Image.fromarray(colour_layout)
    colour_layout.save(out_path)


def main(args):
    pretrained_t2i_model = os.path.join(args.model_weights_dir, "PixArt-XL-2-512x512")
    mini_transformer_dir = os.path.join(args.model_weights_dir, "TIDE_MiniTransformer")
    tide_weight_dir = os.path.join(args.model_weights_dir, "TIDE_r32_64_b4_200k")

    generator = torch.manual_seed(50)
    palette = np.array([[0, 0, 0]] + USIS10K_COLORS_PALETTE)

    # model definitions
    transformer = PixArtSpecialAttnTransformerModel.from_pretrained(
        pretrained_t2i_model,
        subfolder="transformer", torch_dtype=torch.float16
    )
    transformer.requires_grad_(False)

    depth_transformer = MiniTransformerModel.from_config(
        mini_transformer_dir,
        subfolder="mini_transformer",
        torch_dtype=torch.float16
    )
    _state_dict = torch.load(
        os.path.join(mini_transformer_dir, 'mini_transformer/diffusion_pytorch_model.pth'),
        map_location='cpu'
    )
    depth_transformer.load_state_dict(_state_dict)
    depth_transformer.half()
    depth_transformer.requires_grad_(False)
    del _state_dict

    mask_transformer = copy.deepcopy(depth_transformer)

    image_transformer = PeftModel.from_pretrained(
        transformer, os.path.join(tide_weight_dir, 'image_transformer_lora')
    )
    depth_transformer = PeftModel.from_pretrained(
        depth_transformer, os.path.join(tide_weight_dir, 'depth_transformer_lora')
    )
    mask_transformer = PeftModel.from_pretrained(
        mask_transformer, os.path.join(tide_weight_dir, 'mask_transformer_lora')
    )

    tan_modules = TIDE_TANs.from_pretrained(
        os.path.join(tide_weight_dir, 'tan_modules'), torch_dtype=torch.float16
    )

    tide_transformer = TIDETransformerModel(image_transformer, depth_transformer, mask_transformer, tan_modules)
    del image_transformer, depth_transformer, mask_transformer, tan_modules

    model = TIDEPipeline.from_pretrained(
        pretrained_t2i_model,
        transformer=tide_transformer,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # generate image, depth map, semantic mask
    target_image, depth_image, mask_image = model(
        prompt=args.text_prompt,
        num_inference_steps=20,
        generator=generator,
        guidance_scale=2.0,
    )
    target_image = target_image.images[0]
    depth_image = depth_image.images[0]
    mask_image = mask_image.images[0]

    target_image.save(os.path.join(args.output, "image.jpg"))

    depth_image = np.mean(depth_image, axis=-1)
    vis_depth_image = (cmap(depth_image) * 255).astype(np.uint8)
    Image.fromarray(vis_depth_image).save(os.path.join(args.output, "depth.png"))

    id_map = mask_postprocess(mask_image, palette)
    colour_and_vis_id_map(id_map, palette[1:], os.path.join(args.output, "mask.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights_dir", type=str, default="./model_weights")
    parser.add_argument('--text_prompt', type=str, default="A large school of fish swimming in a circle.")
    parser.add_argument('--output', type=str, default="./outputs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    main(args)