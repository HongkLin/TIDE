import numpy as np

from .dataset_palette import (
    USIS10K_COLORS_PALETTE,
)

class IDColour:
    def __init__(self, max_id_num=8, color_palette=USIS10K_COLORS_PALETTE):
        palette = color_palette
        self.palette = np.array(palette)
        self.background_id = 255

    def __call__(self, id_map):
        id_map = np.array(id_map) - 1
        ids = np.unique(id_map)
        valid_ids = np.delete(ids, np.where(ids == self.background_id))

        mask_pixel_values = np.zeros((id_map.shape[0], id_map.shape[1], 3), dtype=np.uint8)
        for id in valid_ids:
            mask_pixel_values[id_map == id, :] = self.palette[id].reshape(1, 3)
        return mask_pixel_values

    def generate_rgb_values(self, num_values):
        step = (255 ** 3) / (num_values - 1)

        rgb_values = []

        for i in range(num_values):
            value_i = round(i * step)
            red_i = value_i // (256 ** 2)
            green_i = (value_i % (256 ** 2)) // 256
            blue_i = value_i % 256
            rgb_values.append((red_i, green_i, blue_i))

        return rgb_values