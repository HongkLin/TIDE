# [CVPR 2025]  A Unified Image-Dense Annotation Generation Model for Underwater Scenes 


[![Website](asset/docs/badge-website.svg)](https://hongklin.github.io/TIDE/)
[![arXiv](https://img.shields.io/badge/Arxiv-2503.21771-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.21771)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

## 🌊 **Introduction** 
We present TIDE, a unified underwater image-dense annotation generation model. Its core lies in the shared layout information and the natural complementarity between multimodal features. Our model, derived from the pre-trained text-to-image model and fine-tuned with underwater data, enables the generation of highly consistent underwater image-dense annotations from solely text conditions.

![TIDE_demo.](asset/images/teasor.png)
---
## 🐚 **News**
- 2025-3-28: The training and inference code is now available!
- 2025-2-27: Our TIDE is accepted to CVPR 2025!
---

## 🪸 Dependencies and Installation

- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.1+cu11.7](https://pytorch.org/)
```bash
conda create -n TIDE python=3.9
conda activate TIDE
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/HongkLin/TIDE
cd TIDE
pip install -r requirements.txt
```

## 🐬 Inference
Download the pre-trained [PixArt-α](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512), [MiniTransformer](https://github.com/HongkLin/TIDE/releases/download/model_weights/TIDE_MiniTransformer.zip), and [TIDE checkpoint](https://github.com/HongkLin/TIDE/releases/download/model_weights/TIDE_r32_64_b4_200k.zip), then modify the model weights path.
```bash
python inference.py --model_weights_dir ./model_weights --text_prompt "A large school of fish swimming in a circle." --output ./outputs
```

## 🐢 Training

### 🏖️ ️Training Data Prepare
- Download [SUIM](https://github.com/xahidbuffon/SUIM), [UIIS](https://github.com/LiamLian0727/WaterMask), [USIS10K](https://github.com/LiamLian0727/USIS10K) datasets. 
- The semantic segmentation annotations are obtained by merging instances with the same semantics.
- The depth annotations are obtained by [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), and the inverse depth results are saved as npy files.
- The image caption and JSON file for organizing training data can follow [Atlantis](https://github.com/zkawfanx/Atlantis), which we also do.

The final dataset should be ordered as follow:
```
datasets/
    UWD_triplets/
        images/
            train_05543.jpg
            ...
        semseg_annotations/
            train_05543.jpg
            ...
        depth_annotations/
            train_05543_raw_depth_meter.npy
            ...
        TrainTIDE_Caption.json
```
If you have prepared the training data and environment, you can run the following script to start the training:
```bash
accelerate launch --num_processes=4 --main_process_port=36666 ./tide/train_tide_hf.py \
--max_train_steps 200000 --learning_rate=1e-4 --train_batch_size=1 \
--gradient_accumulation_steps=1 --seed=42 --dataloader_num_workers=4 --validation_steps 10000 \
--wandb_name=tide_r32_64_b4_200k --output_dir=./outputs/tide_r32_64_b4_200k
```
    
# 🤗Acknowledgements
- Thanks to [Diffusers](https://github.com/huggingface/diffusers) for their wonderful technical support and awesome collaboration!
- Thanks to [Hugging Face](https://github.com/huggingface) for sponsoring the nicely demo!
- Thanks to [DiT](https://github.com/facebookresearch/DiT) for their wonderful work and codebase!
- Thanks to [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) for their wonderful work and codebase!

## 📖BibTeX
```
@inproceedings{lin2025tide,
      title={A Unified Image-Dense Annotation Generation Model for Underwater Scenes}, 
      author={Lin, Hongkai and Liang, Dingkang and Qi, Zhenghao and Bai, Xiang},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2025},
}
```