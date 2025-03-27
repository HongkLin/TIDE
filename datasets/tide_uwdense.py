import os

import datasets
import pandas as pd
import numpy as np

_VERSION = datasets.Version("0.0.1")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "text": datasets.Value("string"),
        "image": datasets.Image(),
        "depth_image": datasets.Value("string"),
        "semantic_image": datasets.Image(),
    },
)
_root = os.getenv("DETECTRON2_DATASETS", "path_to/datasets")
METADATA_PATH = os.path.join(_root, "UWD_triplets/UWDense.json")
IMAGES_DIR = os.path.join(_root, "UWD_triplets/images")
DEPTH_IMAGES_DIR = os.path.join(_root, "UWD_triplets/disparity_depth")
SEMANTIC_IMAGES_DIR = os.path.join(_root, "UWD_triplets/annotations")

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)

class Depth2Underwater(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_path = METADATA_PATH
        images_dir = IMAGES_DIR
        depth_images_dir = DEPTH_IMAGES_DIR
        semantic_images_dir = SEMANTIC_IMAGES_DIR
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": images_dir,
                    "depth_images_dir": depth_images_dir,
                    "semantic_images_dir": semantic_images_dir,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, depth_images_dir, semantic_images_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            text = row["text"]

            image_path = row["image"]
            image_path = os.path.join(images_dir, image_path)
            image = open(image_path, "rb").read()

            if '.jpg' in row["conditioning_image"]:
                depth_image_path = row["conditioning_image"].replace('.jpg', '_raw_depth_meter.npy')
                semantic_image_path = row["conditioning_image"].replace('.jpg', '.png')
            else:
                assert '.png' in row["conditioning_image"]
                depth_image_path = row["conditioning_image"].replace('.png', '_raw_depth_meter.npy')
                semantic_image_path = row["conditioning_image"]
            depth_image_path = os.path.join(
                depth_images_dir, depth_image_path
            )
            semantic_image_path = os.path.join(
                semantic_images_dir, semantic_image_path
            )
            semantic_image = open(semantic_image_path, "rb").read()

            yield row["image"], {
                "text": text,
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "depth_image": depth_image_path,
                "semantic_image": semantic_image
            }