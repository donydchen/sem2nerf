from matplotlib.pyplot import contour
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from utils.data_utils import mask_to_binary_masks, binary_masks_to_contour, contour_to_dist
from torchvision.utils import save_image
from utils.common import convert_mask_label_to_visual


class ImagesDataset3D(Dataset):

    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None, 
                 paths_conf=None):
        if paths_conf is None:
            self.source_paths = sorted(data_utils.make_dataset(source_root))
            self.target_paths = sorted(data_utils.make_dataset(target_root))
        else:
            with open(paths_conf, 'r') as f:
                paths = f.readlines()
                paths = [x.strip() for x in paths if x.strip()]
            src_ext = os.path.splitext(os.listdir(source_root)[0])[-1]
            self.source_paths = [os.path.join(source_root, "%s%s" % (x, src_ext)) for x in paths]
            tar_ext = os.path.splitext(os.listdir(target_root)[0])[-1]
            self.target_paths = [os.path.join(target_root, "%s%s" % (x, tar_ext)) for x in paths]

        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts
        # need to load all gt pose for ray casting
        self.env_name = opts.pigan_curriculum_type
        if self.env_name == "CelebAMask_HQ":
            pose_file_name = "CelebAMask-HQ-pose-anno.txt"
        elif self.env_name == "CatMask":
            pose_file_name = "pose-pseudo.txt"
        else:
            raise Exception("Cannot find environment %s" % self.env_name)
        pose_path = os.path.join(*source_root.split('/')[:-1], pose_file_name)
        self.pose_dict = self.build_poses_mapping_dict(pose_path)

        if opts.use_contour:  # remove the [-2] item, [ToOneHot]
            self.contour_transfrom = transforms.Compose(self.source_transform.transforms[:-2] + [self.source_transform.transforms[-1]])

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        # read the input semantic mask labels
        from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
        from_im = self.update_labels(from_im)
        from_im_raw = from_im
        if self.source_transform:
            from_im = self.source_transform(from_im)

        # read the GT output RGB images
        to_path = self.target_paths[index]
        if not os.path.exists(to_path):  # to cope with edited mask, which does not have GT output
            tmp_name, tmp_ext = os.path.splitext(os.path.basename(to_path))
            tmp_name = tmp_name.split('_')[0] + tmp_ext
            to_path = os.path.join(*to_path.split('/')[:-1], tmp_name)
        to_im = Image.open(to_path).convert('RGB')
        if self.target_transform:
            to_im = self.target_transform(to_im)

        # create augmented data on-the-fly if needed
        if self.opts.use_contour:
            # create contour tensor
            in_mask_np = mask_to_binary_masks(from_im_raw)
            from_contour = binary_masks_to_contour(in_mask_np)
            contour_tensor = self.contour_transfrom(from_contour)

            # create distance data
            dist = contour_to_dist(from_contour)
            dist_tensor = self.contour_transfrom(dist)

            from_im = torch.cat([from_im, contour_tensor, dist_tensor], dim=0)
            # self.vis_input_for_debuging(from_im_raw, contour_tensor, dist_tensor)

        im_pose = self.get_pose_by_image_path(to_path)
        out_dict = {
            'from_im': from_im,
            'to_im': to_im,
            'im_pose': im_pose
        }

        return out_dict

    def build_poses_mapping_dict(self, poses_file):
        if self.env_name == "CelebAMask_HQ":
            n_headlines = 2
            n_pose_params = 3
        elif self.env_name == "CatMask":
            n_headlines = 1
            n_pose_params = 2
        else:
            raise Exception("Cannot find environment %s" % self.env_name)

        with open(poses_file, 'r') as f:
            lines = f.readlines()
            lines = [x for x in lines if x.strip()][n_headlines:]
        poses_dict = {}
        for line in lines:
            items = line.split(' ')
            cur_id = int(items[0].split('.')[0])
            poses_dict[cur_id] = [float(x) for x in items[1:n_pose_params+1]]

        return poses_dict

    def get_pose_by_image_path(self, image_path):
        cur_id = int(os.path.basename(image_path).split('.')[0])
        cur_pose = self.pose_dict[cur_id]
        if self.env_name == "CelebAMask_HQ":
            yaw = np.deg2rad(90 - cur_pose[0])
            pitch = np.deg2rad(90 + cur_pose[1])
        elif self.env_name == "CatMask":
            yaw = cur_pose[0]
            pitch = cur_pose[1]
        else:
            raise Exception("Cannot find environment %s" % self.env_name)
        return yaw, pitch

    def update_labels(self, ori_labels):
        if not self.opts.use_merged_labels:
            return ori_labels
        
        if self.env_name == "CelebAMask_HQ":
            new_labels = [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        elif self.env_name == "CatMask":
            new_labels = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 7, 0]
        else:
            raise Exception("Cannot find environment %s" % self.env_name)

        labels_np = np.array(ori_labels)
        for idx, label in enumerate(new_labels):
            labels_np[labels_np == idx] = label
        new_labels = Image.fromarray(np.uint8(labels_np))
        return new_labels

    def vis_input_for_debuging(self, from_im_raw, contour_tensor, dist_tensor):
        tmp = transforms.Compose(self.source_transform.transforms[:-2] + [transforms.PILToTensor()])(from_im_raw)
        vis_from_im = convert_mask_label_to_visual(tmp, 'CelebAMask_HQ', label_nc=16)[0]
        save_image(vis_from_im, 'image.png', normalize=True)
        save_image(contour_tensor, 'contour.png')
        save_image(dist_tensor, 'dist.png')
        assert False
