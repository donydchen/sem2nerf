import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import skvideo.io
from PIL import Image
import math
import torch
from torch.utils.data import DataLoader
import sys
from torchvision.utils import save_image
import torch.nn.functional as F
import cv2

sys.path.append(".")
sys.path.append("..")
from configs import data_configs
from datasets.inference_dataset import InferenceDataset3D
from options.test_options import TestOptions
from models.sem2nerf import Sem2NeRF
from models.pigan.op import curriculums
from utils.common import convert_mask_label_to_visual


def run():
    test_opts = TestOptions().parse()
    loaded_iter = os.path.splitext(os.path.basename(test_opts.checkpoint_path))[0].split('_')[-1]

    out_path_results = os.path.join(test_opts.exp_dir, 'iter_%s_infer_results_%d' % 
                                        (loaded_iter, test_opts.test_output_size))
    out_path_videos = os.path.join(test_opts.exp_dir, 'iter_%s_infer_videos_%d' % 
                                        (loaded_iter, test_opts.test_output_size))

    os.makedirs(out_path_results, exist_ok=True)
    if test_opts.render_videos:
        os.makedirs(out_path_videos, exist_ok=True)
    with open(os.path.join(test_opts.exp_dir, 'inference.bash'), 'a+') as f:
        f.write("python %s\n" % ' '.join(sys.argv))

    # define img / video name postfix
    if test_opts.latent_mask is not None:
        name_postfix = '_mask' + test_opts.latent_mask.replace(',', '') \
                        + '_seed' + str(test_opts.inject_code_seed)
    else:
        name_postfix = ''

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 128
    opts['device'] = 'cuda:0'
    opts = Namespace(**opts)
    opts.gpu_ids = [int(x) for x in opts.gpu_ids.split(',')]

    net = Sem2NeRF(opts)
    net.eval()
    net.set_devices(opts.gpu_ids, mode='test')

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms(opts.output_size, opts.src_input_size)
    dataset = InferenceDataset3D(root=opts.data_path,
                                 transform=transforms_dict['transform_inference'],
                                 opts=opts, paths_conf=opts.infer_paths_conf)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=opts.use_shuffle,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    if opts.render_videos:
        trajectory = get_trajectory(opts.trajectory, getattr(curriculums, opts.pigan_curriculum_type), opts)

    if opts.latent_mask is not None:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        torch.manual_seed(opts.inject_code_seed)
        vec_to_inject = torch.randn((1, net.decoder.z_dim)).cuda()
        with torch.no_grad():
            latent_to_inject = net.decoder.siren.mapping_network(vec_to_inject)
            batch_latent_to_inject = (latent_to_inject[0].repeat(opts.test_batch_size, 1),
                                    latent_to_inject[1].repeat(opts.test_batch_size, 1))
    else:
        latent_mask = None
        latent_to_inject = None
        batch_latent_to_inject = None

    global_i = 0
    global_time = []
    for input_batch in dataloader:
        print("[%d / %d] Rendering..." % (global_i, min(len(dataset), opts.n_images)))
        if global_i >= opts.n_images:
            break
        
        with torch.no_grad():
            tic = time.time()
            # reset fov to original
            net.pigan_curriculum['fov'] = 12 if opts.pigan_curriculum_type == 'CelebAMask_HQ' else 18
            print('Reset fov to', net.pigan_curriculum['fov'])
            result_batch = run_on_batch_images(input_batch, net, opts, latent_mask, 
                                               latent_to_inject=batch_latent_to_inject)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            images = result_batch[i]
            im_id = input_batch[-1][i]
            save_image(images, os.path.join(out_path_results, f'{im_id}{name_postfix}.png'), normalize=True)

            if opts.label_nc > 1 and opts.use_original_pose and opts.mask_overlay_weight > 0.:
                mask_im = (images[0].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.
                rgb_im = (images[1].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.
                mask_im = cv2.cvtColor(mask_im.astype(np.uint8), cv2.COLOR_RGB2BGR)
                rgb_im = cv2.cvtColor(rgb_im.astype(np.uint8), cv2.COLOR_RGB2BGR)
                overlay_im = cv2.addWeighted(rgb_im, 1. - opts.mask_overlay_weight,
                                                mask_im, opts.mask_overlay_weight, 0)
                cv2.imwrite(os.path.join(out_path_results, f'{im_id}{name_postfix}_overlay.png'), overlay_im)

            if opts.render_videos:
                writer = skvideo.io.FFmpegWriter(os.path.join(out_path_videos, f'{im_id}{name_postfix}.mp4'), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21', '-vf': 'setpts=1.5*PTS'})
                cur_input = input_batch[0][i].unsqueeze(0)
                with torch.no_grad():
                    frames = render_video_per_scene(cur_input, net, opts, trajectory,
                                                    latent_mask, latent_to_inject)
                frames = [tensor_to_PIL(x) for x in frames[0]]
                for frame in frames:
                    writer.writeFrame(np.array(frame))
                writer.close()

            global_i += 1

    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)


def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


def get_trajectory(trajectory_type, curriculum, opts):
    trajectory = []
    if trajectory_type == 'front':
        for t in np.linspace(0, 1, opts.num_frames):
            pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
            yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
            fov = 12
            fov = 12 + 5 + np.sin(t * 2 * math.pi) * 5
            trajectory.append((pitch, yaw, fov))
    elif trajectory_type == 'orbit':
        for t in np.linspace(0, 1, opts.num_frames):
            pitch = math.pi/4
            yaw = t * 2 * math.pi
            fov = curriculum['fov']
            trajectory.append((pitch, yaw, fov))
    return trajectory


def run_on_batch_images(inputs, net, opts, latent_mask=None, latent_to_inject=None):
    result_batch = []
    poses = []
    init_h_mean = getattr(curriculums, opts.pigan_curriculum_type)['h_mean']
    v_mean = getattr(curriculums, opts.pigan_curriculum_type)['v_mean']

    if opts.use_original_pose:
        x, pose, _ = inputs
        # v_mean = pose[1]
        poses.append(pose)
    else:
        x, _ = inputs

    v_mean = torch.tensor(v_mean).repeat(x.shape[0])
    x = x.float().to(opts.device)

    # prepare all poses
    if opts.pigan_curriculum_type == 'CelebAMask_HQ':
        face_angles = [-0.45, -0.25, 0., 0.25, 0.45]
    else: 
        face_angles = [-0.4, -0.2, 0., 0.2, 0.4]

    face_angles = [a + init_h_mean for a in face_angles]
    for face_angle in face_angles:
        cur_pose = (torch.tensor(face_angle).repeat(x.shape[0]), v_mean)
        poses.append(cur_pose)

    # update x based on type and size
    x_im = x.clone().cpu().detach()
    if opts.label_nc > 1:
        x_im = torch.argmax(x_im, dim=1).int()
        x_im = convert_mask_label_to_visual(x_im, env_name=opts.pigan_curriculum_type, label_nc=opts.label_nc)
    if opts.src_input_size != opts.test_output_size:
        x_im = F.interpolate(x_im, size=opts.test_output_size)
    result_batch.append(x_im)

    for cur_pose in tqdm(poses, desc="Novel View Generating..."):
        if latent_mask is None:
            y_hat = net.forward_inference(x, cur_pose, return_latents=False)['images']
        else:
            # inject the origin latent code, so as to change style
            y_hat = net.forward_inference(x, cur_pose, return_latents=False,
                                latent_mask=latent_mask, 
                                inject_latent=latent_to_inject)['images']
        result_batch.append(y_hat.cpu().detach())

    result_batch = torch.stack(result_batch, dim=1)  # B, N, C, H, W

    return result_batch


def render_video_per_scene(input_image, net, opts, trajectory, latent_mask=None, latent_to_inject=None):
    result_batch = []
    x = input_image.float().to(opts.device)
    for pitch, yaw, fov in tqdm(trajectory, desc="Video Generating..."):
        pitch = torch.tensor(pitch).repeat(x.shape[0])
        yaw = torch.tensor(yaw).repeat(x.shape[0])
        net.pigan_curriculum['fov'] = fov
        cur_pose = (yaw, pitch)
        if latent_mask is None:
            y_hat = net.forward_inference(x, cur_pose, return_latents=False)['images']
        else:
            y_hat = net.forward_inference(x, cur_pose, return_latents=False, latent_mask=latent_mask,
                                          inject_latent=latent_to_inject)['images']
        result_batch.append(y_hat.cpu().detach())

    result_batch = torch.stack(result_batch, dim=1)  # B, N, C, H, W
    return result_batch


if __name__ == '__main__':
    run()
