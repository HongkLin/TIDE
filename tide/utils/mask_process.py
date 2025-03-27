import numpy as np
import torch
from PIL import Image
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian



def apply_crf(probability_matrix, num_classes):
    h, w = probability_matrix.shape[-2:]

    dcrf_model = dcrf.DenseCRF(h * w, num_classes)

    unary = unary_from_softmax(probability_matrix)
    unary = np.ascontiguousarray(unary)
    dcrf_model.setUnaryEnergy(unary)

    feats = create_pairwise_gaussian(sdims=(10, 10), shape=(h, w))

    dcrf_model.addPairwiseEnergy(feats, compat=3,
                                 kernel=dcrf.DIAG_KERNEL,
                                 normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = dcrf_model.inference(5)

    MAP = np.argmax(Q, axis=0).reshape((h, w))
    return MAP


def colour_and_vis_id_map(id_map, palette, out_path):
    palette = np.array(palette)

    id_map = id_map.astype(np.uint8)
    id_map -= 1
    ids = np.unique(id_map)
    valid_ids = np.delete(ids, np.where(ids == 255))

    colour_layout = np.zeros((id_map.shape[0], id_map.shape[1], 3), dtype=np.uint8)
    for id in valid_ids:
        colour_layout[id_map == id, :] = palette[id].reshape(1, 3)
    colour_layout = Image.fromarray(colour_layout)
    colour_layout.save(out_path)


def compute_distance(matrix_A, matrix_B):
    matrix_A = matrix_A[:, np.newaxis, :]

    diff_squared = (matrix_A - matrix_B) ** 2

    distances = np.sqrt(np.sum(diff_squared, axis=2))
    return distances


def rgb2id(matrix_A, matrix_B, th=None, return_crf_refine=True, t=40):
    h, w = matrix_A.shape[:2]
    matrix_A = matrix_A.reshape(-1, 3)

    distances = compute_distance(matrix_A, matrix_B)
    min_distance_indices = np.argmin(distances, axis=1)

    if th is not None:
        min_distances = np.min(distances, axis=1)
        min_distance_indices[min_distances > th] = len(matrix_B) - 1

    crf_results = None
    if return_crf_refine:
        prob = 1 - distances / distances.sum(1)[:, None]
        prob = torch.tensor(prob).reshape(h, w, -1).permute(2, 0, 1)
        prob = torch.nn.functional.softmax(t * prob, dim=0).numpy()
        crf_results = apply_crf(prob, num_classes=len(matrix_B))
    return min_distance_indices.reshape(w, h), crf_results


def filter_small_regions(arr, min_area=100, ignore_class=0):
    unique_labels = np.unique(arr)
    result = arr.copy()

    for label in unique_labels:
        if label == ignore_class:
            continue
        binary_image = np.where(arr == label, 1, 0).astype('uint8')
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                result[labels == i] = ignore_class

    return result


def mask_postprocess(mask_image, palette, ignore_class=0):
    mask_image = np.array(mask_image)
    _, refine_id_map = rgb2id(mask_image, palette)
    id_map = filter_small_regions(refine_id_map)
    id_map[id_map == 7] = ignore_class # ignore sea-floor
    return id_map