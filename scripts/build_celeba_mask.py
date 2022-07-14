import glob
import os
import tqdm
from PIL import Image
import numpy as np


# Using label mapping from https://github.com/switchablenorms/CelebAMask-HQ/blob/master/MaskGAN_demo/demo.py
TYPE_TO_LABEL = {
    'skin': 1,
    'nose': 2,
    'eye_g': 3,
    'l_eye': 4,
    'r_eye': 5,
    'l_brow': 6,
    'r_brow': 7, 
    'l_ear': 8,
    'r_ear': 9,
    'mouth': 10,
    'u_lip': 11,
    'l_lip': 12,
    'hair': 13,
    'hat': 14,
    'ear_r': 15,
    'neck_l': 16,
    'neck': 17,
    'cloth': 18
}


def get_mask_type(image_path):
    return '_'.join(os.path.splitext(os.path.basename(image_path))[0].split('_')[1:])


def build_mask_mapping_dict(mask_folder):
    # build a dictionary with orig_id
    mask_paths = []
    for sub_fold in glob.glob(os.path.join(mask_folder, '*/')):
        mask_paths += list(glob.glob(os.path.join(sub_fold, '*.png')))
    map_dict = {}
    for mask_path in mask_paths:
        basename = os.path.basename(mask_path)
        cur_id = int(basename.split('_')[0])
        if cur_id not in map_dict:
            map_dict[cur_id] = []
        map_dict[cur_id].append(mask_path)
    return map_dict


def main(in_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    map_dict = build_mask_mapping_dict(in_dir)
    for img_id, mask_paths in tqdm.tqdm(map_dict.items(), desc='Building Masks'):
        # if img_id != 419: 
            # continue 
        # convert the single file label image to one mask with one value for each label
        combined_mask = []
        for mask_path in mask_paths:
            cur_mask = np.array(Image.open(mask_path).convert('P'))
            cur_mask[cur_mask == 225] = TYPE_TO_LABEL[get_mask_type(mask_path)]
            combined_mask.append(cur_mask)
        combined_mask = np.stack(combined_mask, axis=-1)
        combined_mask = np.max(combined_mask, axis=-1)

        mask_im = Image.fromarray(combined_mask.astype(np.uint8))
        mask_im.save(os.path.join(out_dir, "%s.png" % img_id))

    print("All Done!")


if __name__ == "__main__":
    in_dir = 'data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    out_dir = 'data/CelebAMask-HQ/masks'
    main(in_dir, out_dir)
