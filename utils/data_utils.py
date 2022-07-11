"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
import cv2
import numpy as np
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def mask_to_binary_masks(in_mask):
    '''
        in_mask: PIL.Image
        binary_masks: list of numpy binary masks, ignore 0 (background)
    '''
    in_mask_np = np.array(in_mask)
    labels = np.unique(in_mask_np)[1:]
    binary_masks = []
    for label in labels:
        binary_masks.append((in_mask_np == label).astype(in_mask_np.dtype))
    return binary_masks


def binary_masks_to_contour(binary_masks):
    h, w = binary_masks[0].shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for bin_mask in binary_masks:
        cnts = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(mask, [c], -1, (255, 255, 255), thickness=3)
    contour = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contour = Image.fromarray(contour)
    return contour


def contour_to_dist(contour):
    invert_contour = cv2.bitwise_not(np.array(contour))
    dist = cv2.distanceTransform(invert_contour, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    dist = dist * 255
    dist = Image.fromarray(dist.astype(np.uint8))
    return dist
