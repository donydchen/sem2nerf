"""
This file runs the main training/val loop
"""
import pprint
import sys

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach3d import Coach3D
from utils import distri_utils
import os
import json
import yaml


def main():
    opts = TrainOptions().parse()

    # init distributed training if needed
    distri_utils.init_distributed_mode(opts)
    opts.distributed_train = opts.distributed

    if (not opts.distributed_train) and (not 'debug' in opts.exp_dir) and (os.path.exists(opts.exp_dir)):
        raise Exception('Oops... {} already exists'.format(opts.exp_dir))
    distri_utils.fn_on_master_if_distributed(opts.distributed_train, os.makedirs,
                                             opts.exp_dir, exist_ok=('debug' in opts.exp_dir))

    # update gpu_ids
    opts.gpu_ids = [int(x) for x in opts.gpu_ids.split(',')]
    if -1 in opts.gpu_ids:
        opts.gpu_ids = ['cpu']

    # update data loading for gt image with regard to pigan_step configs
    with open(opts.pigan_steps_conf, 'r') as f:
        train_steps_env = yaml.load(f, Loader=yaml.Loader)[0]
    gt_img_size = train_steps_env['img_size'] if opts.patch_train else train_steps_env['resolution_vol']
    opts.output_size = gt_img_size
    if gt_img_size <= 256:
        opts.image_load_size = 320
        opts.image_crop_size = 256
    elif gt_img_size == 512:
        opts.image_load_size = 640
        opts.image_crop_size = 512
    else:
        raise Exception('Undefined img_size [%d], use either <=256 or 512' % train_steps_env['img_size'])

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)

    if opts.distributed_train:
        if distri_utils.is_main_process():
            with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
                json.dump(opts_dict, f, indent=4, sort_keys=True)
            with open(os.path.join(opts.exp_dir, 'train.bash'), 'w') as f:
                prefix = 'python -m torch.distributed.launch --nproc_per_node=%d' % distri_utils.get_world_size()
                f.write("%s %s\n" % (prefix, ' '.join(sys.argv)))
    else:
        with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
            json.dump(opts_dict, f, indent=4, sort_keys=True)
        with open(os.path.join(opts.exp_dir, 'train.bash'), 'w') as f:
            f.write("python %s\n" % ' '.join(sys.argv))

    coach = Coach3D(opts)
    coach.train()


if __name__ == '__main__':
    main()
