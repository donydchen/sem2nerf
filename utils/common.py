import PIL
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pyparsing import Or
import torch
from collections import OrderedDict


# Log images
def log_input_image(x, opts):
    if opts.label_nc == 0:
        return tensor2im(x)
    else:
        return tensor2map(x, opts.pigan_curriculum_type, opts.label_nc)


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def tensor2map(var, env_name, label_nc):
    mask = np.argmax(var.data.cpu().numpy(), axis=0)
    colors = get_colors(env_name, label_nc)
    mask_image = np.ones(shape=(mask.shape[0], mask.shape[1], 3))
    for class_idx in np.unique(mask):
        mask_image[mask == class_idx] = colors[class_idx]
    mask_image = mask_image.astype('uint8')
    return Image.fromarray(mask_image)


# Visualization utils
def get_colors(env_name, label_nc):
    if env_name == "CelebAMask_HQ":
        # currently support up to 19 classes (for the celebs-hq-mask dataset)
        if label_nc == 19:
            colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
                    [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
                    [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
        # [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        elif label_nc == 16:
            colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [0, 255, 255],
                      [255, 204, 204], [102, 51, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
                      [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
        else:
            raise Exception('Unknown label number: %d' % label_nc)
    elif env_name == "CatMask":
        if label_nc == 16:
            colors = [[255, 255, 255], [220, 220, 0], [190, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35],
                    [102, 102, 156], [152, 251, 152], [119, 11, 32], [244, 35, 232], [220, 20, 60], [52, 83, 84],
                    [194, 87, 125], [225,  96, 18], [31, 102, 211], [104, 131, 101]]
        elif label_nc == 8:
            colors = [[0, 0, 0], [204, 0, 0], [102, 51, 0], [0, 255, 255], [255, 204, 204],
                      [25, 204, 204], [76, 153, 0], [0, 0, 0]]
        else:
            raise Exception('Unknown label number: %d' % label_nc)
    else:
        raise Exception("Cannot find environment %s" % env_name)
    return colors


def vis_faces(log_hooks):
    display_count = len(log_hooks)
    img_keys = [k for (k, v) in log_hooks[0].items() if isinstance(v, Image.Image)]
    fig = plt.figure(figsize=(3 * len(img_keys) - 1, 4 * display_count))
    gs = fig.add_gridspec(display_count, len(img_keys))
    for i in range(display_count):
        hooks_dict = log_hooks[i]
        fig.add_subplot(gs[i, 0])
        if 'diff_input' in hooks_dict:
            vis_faces_with_id(hooks_dict, fig, gs, i)
        else:
            vis_faces_no_id(hooks_dict, fig, gs, i)
        other_keys = [k for k in img_keys if k not in ['input_face', 'target_face', 'output_face']]
        for idx, k in enumerate(other_keys):
            fig.add_subplot(gs[i, 3 + idx])
            plt.imshow(hooks_dict[k])
            plt.title(k.split('_')[-1])
    plt.tight_layout()
    return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input_face'])
    plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['target_face'])
    plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
                                                     float(hooks_dict['diff_target'])))
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))


def vis_faces_no_id(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input_face'], cmap="gray")
    plt.title('Input')
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['target_face'])
    plt.title('Target')
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output')


def convert_mask_label_to_visual(mask_label, env_name, normalize=True, label_nc=19):
    """
    Batch masks based
    Inverse version of convert_mask_visual_to_label
        Input: mask_label, size: BxHxW, dtype: long
        Return: mask_visual, size: BxCxHxW, dtype: float 
        Inspired by: https://discuss.pytorch.org/t/visualise-the-test-images-after-training-the-model-on-segmentation-task/56615/4
    """
    # device = mask_label.get_device()
    colors = get_colors(env_name, label_nc)
    mask_mapping = {}
    for idx, color in enumerate(colors):
        # print(idx)
        mask_mapping[tuple(float(x) / 255. for x in color)] = idx
    mask_visual = torch.zeros_like(mask_label).unsqueeze(-1).repeat(1, 1, 1, 3).float()
    for k, v in mask_mapping.items():
        # print('currrent v', v)
        mask_visual[mask_label == v, :] = torch.tensor(k).float().view(1, 3)

    mask_visual = mask_visual.permute(0, 3, 1, 2)
    if normalize:
        mask_visual = (mask_visual - 0.5) / 0.5
    return mask_visual


def filt_ckpt_keys(ckpt, item_name, model_name):
    # if item_name in ckpt:
    assert item_name in ckpt, "Cannot find [%s] in the checkpoints." % item_name
    d = ckpt[item_name]
    d_filt = OrderedDict()
    for k, v in d.items():
        k_list = k.split('.')
        if k_list[0] == model_name:
            if k_list[1] == 'module':
                d_filt['.'.join(k_list[2:])] = v
            else:
                d_filt['.'.join(k_list[1:])] = v
    return d_filt
