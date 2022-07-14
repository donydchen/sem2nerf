from training.ranger import Ranger
from models.sem2nerf import Sem2NeRF
from models.discriminators import Discriminator
from criteria.lpips.lpips import LPIPS
from datasets.images_dataset import ImagesDataset3D
from configs import data_configs
from criteria import w_norm
from utils import common, train_utils, distri_utils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
from time import time
from timm.scheduler.cosine_lr import CosineLRScheduler
import random
import numpy as np


matplotlib.use('Agg')



class Coach3D:
    def __init__(self, opts):
        self.opts = opts
        self.global_step = 0

        # Initialize network
        self.net = Sem2NeRF(self.opts)
        self.net.update_pigan_curriculum()
        if opts.distributed and not opts.use_multi_nodes:
            self.net.set_devices([distri_utils.get_rank()])
            self.device = distri_utils.get_rank()  # opts.gpu
        else:
            self.net.set_devices(self.opts.gpu_ids)
            self.device = self.opts.gpu_ids[0]

        # Initialize loss
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.w_norm_lambda > 0:
            self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()
        optimizers = [self.optimizer]

        # Initialize GAN training if needed
        if self.opts.dis_lambda > 0:
            self.net_dis = Discriminator(self.net.pigan_curriculum[0]['resolution_vol'], self.opts)
            if opts.distributed:
                self.net_dis.set_devices([distri_utils.get_rank()])
            else:
                self.net_dis.set_devices(self.opts.gpu_ids)
            self.optimizer_dis = self.configure_dis_optimizers()
            optimizers.append(self.optimizer_dis)

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        if opts.distributed:
            num_tasks = distri_utils.get_world_size()
            global_rank = distri_utils.get_rank()
            train_sampler = torch.utils.data.DistributedSampler(
                self.train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            test_sampler = torch.utils.data.DistributedSampler(
                self.test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
            test_sampler = torch.utils.data.SequentialSampler(self.test_dataset)

        self.train_dataloader = DataLoader(self.train_dataset,
                                     sampler=train_sampler,
                                     batch_size=self.opts.batch_size,
                                     num_workers=int(self.opts.workers),
                                     drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                    sampler=test_sampler,
                                    batch_size=self.opts.test_batch_size,
                                    num_workers=int(self.opts.test_workers),
                                    drop_last=False)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = distri_utils.fn_on_master_if_distributed(self.opts.distributed_train, SummaryWriter, log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        # update data via checkpoint if continued training
        if self.opts.checkpoint_path is not None and self.opts.continue_training:
            loaded_ckpts = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.global_step = loaded_ckpts['global_step']
            self.best_val_loss = loaded_ckpts['best_val_loss']
            self.optimizer.load_state_dict(loaded_ckpts['optimizer'])
            if self.opts.dis_lambda > 0:
                self.optimizer_dis.load_state_dict(loaded_ckpts['dis_optimizer'])
            print("Continued training, updated global_step to %d, best_val_loss to %.4f, loaded params for optimizers" % (self.global_step, self.best_val_loss))
        # configure lr_scheduler if needed
        if self.opts.scheduler_name != 'none':
            self.lr_schedulers = self.configure_scheduler(optimizers)

    def train(self):
        # show param numbers
        nparams = sum(p.numel() for p in self.net.encoder.parameters())
        print(f">>> [{self.net.subnet_attri_getter('encoder').__class__.__name__}] Parameter numbers: {nparams:,}")
        nparams = sum(p.numel() for p in self.net.subnet_attri_getter('decoder').siren.mapping_network.parameters())
        nparams2 = sum(p.numel() for p in self.net.subnet_attri_getter('decoder').siren.network.parameters())
        print(f">>> [Decoder] Parameter numbers: all {nparams:,} with old_map {nparams2:,}")

        self.net.train()
        self.net.requires_grad(enc_flag=True, dec_flag=self.opts.train_decoder)  # set grad accordingly
        if self.opts.dis_lambda > 0:
            self.net_dis.train()

        # init validate setting
        if self.opts.val_latent_mask is not None:
            self.val_latent_mask = [int(x) for x in self.opts.val_latent_mask.split(',')]
            self.val_inject_latent = self.net.latent_avg
        else:
            self.val_latent_mask = None
            self.val_inject_latent = None

        print('Start training from step %d, max step is %d' % (self.global_step, self.opts.max_steps))
        while self.global_step < self.opts.max_steps:
            for batch in self.train_dataloader:

                # train generator first if applicable
                if self.opts.dis_lambda > 0:
                    self.net.requires_grad(enc_flag=True, dec_flag=self.opts.train_decoder)
                    train_utils.requires_grad(self.net_dis, False)
                    self.net.train()
                    self.net_dis.train()

                self.optimizer.zero_grad()
                x, y, pose = batch['from_im'], batch['to_im'], batch['im_pose']
                x = x.float()
                # apply GAN loss on those extrem pose cases
                train_rand_pose = False
                if random.random() < self.opts.train_rand_pose_prob:
                    train_rand_pose = True
                    pose = self.__get_randn_poses(x.shape[0])
                out_dict = self.net(x, pose, return_latents=True, iter_num=self.global_step)
                y_hat = out_dict['images']
                latent = out_dict['latents']
                sample_pattern = out_dict['sample_pattern']

                # match the batch size and shape
                x = x[:y_hat.shape[0]].to(self.device)
                y = y[:y_hat.shape[0]].to(self.device)
                assert y.shape[-1] >= self.net.cur_pigan_env['img_size'], "Please load larger image, current: %d, expect: %d" % (y.shape[-1], self.net.cur_pigan_env['img_size'])
                if y.shape[-1] != self.net.cur_pigan_env['img_size']:
                    y = F.interpolate(y, size=self.net.cur_pigan_env['img_size'], mode='bilinear', align_corners=True)
                if self.opts.patch_train and sample_pattern is not None:
                    y = F.grid_sample(y, sample_pattern.to(self.device), mode='bilinear', align_corners=True)
                if self.opts.dis_lambda > 0:
                    x = F.interpolate(x, size=y_hat.shape[-1], mode='nearest')

                loss, loss_dict = self.calc_loss(x, y, y_hat, latent, train_rand_pose=train_rand_pose)
                loss.backward()
                self.optimizer.step()
                if self.opts.scheduler_name != 'none':
                    loss_dict['lr'] = float(self.lr_schedulers[0]._get_lr(self.global_step)[0])
                    self.lr_schedulers[0].step_update(self.global_step)

                # train discriminator if needed
                if self.opts.dis_lambda > 0:
                    d_loss_dict = self.train_step_discriminator(x, y, y_hat)
                    loss_dict.update(d_loss_dict)
                    if self.opts.scheduler_name != 'none':
                        loss_dict['lr_dis'] = float(self.lr_schedulers[-1]._get_lr(self.global_step)[0])
                        self.lr_schedulers[-1].step_update(self.global_step)

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(x, y, y_hat, title='images/train/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    distri_utils.fn_on_master_if_distributed(self.opts.distributed_train, self.log_metrics, loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    s_t = time()
                    val_loss_dict = self.validate()

                    self.checkpoint_me(val_loss_dict, is_best=False, save_latest=True)
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)
                    print('Finish Validation. Time taken: %.4fs' % (time() - s_t))

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1

    def validate(self, log_info=True):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            x, y, pose = batch['from_im'], batch['to_im'], batch['im_pose']

            with torch.no_grad():
                x = x.float()
                out_dict = self.net.forward_inference(x, pose, latent_mask=self.val_latent_mask,
                                                        inject_latent=self.val_inject_latent, return_latents=True)
                y_hat, latent = out_dict['images'], out_dict['latents']
                y = y.float().to(self.device)
                if y.shape[-1] != y_hat.shape[-1]:
                    y = F.interpolate(y, size=y_hat.shape[-1])
                _, cur_loss_dict = self.calc_loss(x, y, y_hat, latent, phase='val')
                # add two novel views
                front_pose = (torch.deg2rad(torch.ones_like(pose[0]) * 90), torch.deg2rad(torch.ones_like(pose[1]) * 90))
                y_hat_front = self.net.forward_inference(x, front_pose, latent_mask=self.val_latent_mask,
                                                                    inject_latent=self.val_inject_latent,
                                                                    return_latents=False)['images']
                ym_pose = (torch.deg2rad(180 - torch.rad2deg(pose[0])), pose[1])
                y_hat_yawmirror = self.net.forward_inference(x, ym_pose, latent_mask=self.val_latent_mask,
                                                                    inject_latent=self.val_inject_latent,
                                                                    return_latents=False)['images']
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            postfix = '{:04d}'.format(batch_idx)
            if self.opts.distributed_train:
                x = distri_utils.gather_object_in_order(x.cuda())
                y = distri_utils.gather_object_in_order(y)
                y_hat = distri_utils.gather_object_in_order(y_hat)
                y_hat_front = distri_utils.gather_object_in_order(y_hat_front)
                y_hat_yawmirror = distri_utils.gather_object_in_order(y_hat_yawmirror)
                display_count = self.opts.test_batch_size * distri_utils.get_world_size()
            else:
                display_count = self.opts.test_batch_size
            self.parse_and_log_test_images(x, y, y_hat, y_hat_front, y_hat_yawmirror,
                             title='images/test/faces',
                             subscript=postfix,
                             display_count=display_count)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 8:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        if log_info:
            distri_utils.fn_on_master_if_distributed(self.opts.distributed_train, self.log_metrics, loss_dict, prefix='test')
            self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best, save_latest=False):
        if save_latest:
            save_name = 'latest_model.pt'
        else:
            save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        # torch.save(save_dict, checkpoint_path)
        distri_utils.fn_on_master_if_distributed(self.opts.distributed_train, torch.save, save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                distri_utils.fn_on_master_if_distributed(self.opts.distributed_train, f.write,
                                                         f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
            else:
                distri_utils.fn_on_master_if_distributed(self.opts.distributed_train, f.write,
                                                         f'Step - {self.global_step}, \n{loss_dict}\n')

    def configure_optimizers(self):
        params = list(self.net.encoder.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        elif self.opts.optim_name == 'adamw':
            optimizer = torch.optim.AdamW(params, eps=1e-8, betas=(0.9, 0.999),
                                          lr=self.opts.learning_rate, weight_decay=0.05)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_dis_optimizers(self):
        params = list(self.net_dis.model.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate_dis)
        elif self.opts.optim_name == 'adamw':
            optimizer = torch.optim.AdamW(params, eps=1e-8, betas=(0.9, 0.999),
                                          lr=self.opts.learning_rate_dis, weight_decay=0.05)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate_dis)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
        print(f'Loading dataset for {self.opts.dataset_type}')
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms(self.opts.output_size, self.opts.src_input_size)
        train_dataset = ImagesDataset3D(source_root=dataset_args['train_source_root'],
                                target_root=dataset_args['train_target_root'],
                                source_transform=transforms_dict['transform_source'],
                                target_transform=transforms_dict['transform_gt_train'],
                                opts=self.opts, paths_conf=self.opts.train_paths_conf)
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms(self.opts.output_size, self.opts.src_input_size)
        test_dataset = ImagesDataset3D(source_root=dataset_args['test_source_root'],
                               target_root=dataset_args['test_target_root'],
                               source_transform=transforms_dict['transform_source'],
                               target_transform=transforms_dict['transform_test'],
                               opts=self.opts, paths_conf=self.opts.test_paths_conf)
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat, latent, phase='train', train_rand_pose=False):
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = y_hat.to(self.device)

        loss_dict = {}
        loss = 0.0
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            if train_rand_pose:
                loss_l2 = torch.zeros_like(loss_l2)
            loss_dict['loss_l2'] = loss_l2
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.l1_lambda > 0:
            loss_l1 = F.l1_loss(y_hat, y)
            if train_rand_pose:
                loss_l1 = torch.zeros_like(loss_l1)
            loss_dict['loss_l1'] = loss_l1
            loss += loss_l1 * self.opts.l1_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            if train_rand_pose:
                loss_lpips = torch.zeros_like(loss_lpips)
            loss_dict['loss_lpips'] = loss_lpips
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.w_norm_lambda > 0:
            # reshape to match stylegan latent code structure
            codes, codes_avg = latent
            b = codes[0].shape[0]
            reshaped_latent = torch.cat((codes[0].reshape(b, 9, -1), codes[1].reshape(b, 9, -1)), dim=-1)
            reshaped_latent_avg = torch.cat((codes_avg[0].reshape(b, 9, -1), codes_avg[1].reshape(b, 9, -1)), dim=-1)
            loss_w_norm = self.w_norm_loss(reshaped_latent.to(self.device), reshaped_latent_avg.to(self.device))
            loss_dict['loss_w_norm'] = loss_w_norm
            loss += loss_w_norm * self.opts.w_norm_lambda
        if self.opts.dis_lambda > 0 and phase == 'train':
            if y_hat.shape[-1] != self.net_dis.img_size:
                d_fake = self.net_dis(F.interpolate(y_hat, self.net_dis.img_size, mode='bilinear', align_corners=False),
                                         F.interpolate(x, self.net_dis.img_size, mode='nearest'))
            else:
                d_fake = self.net_dis(y_hat, x)
            loss_g = train_utils.compute_bce(d_fake, 1)
            loss_dict['loss_g'] = loss_g
            loss += loss_g * self.opts.dis_lambda
        loss_dict['loss'] = loss

        # reduce loss value accross all processes for log purpose
        if self.opts.distributed_train:
            loss_dict = distri_utils.reduce_dict(loss_dict, average=True)
        loss_dict = {k: v.item() for k, v in loss_dict.items()}

        return loss, loss_dict

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, x, y, y_hat, title, subscript=None, display_count=2):
        im_data = []
        display_count = min(x.shape[0], display_count)
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            im_data.append(cur_im_data)
        distri_utils.fn_on_master_if_distributed(self.opts.distributed_train, self.log_images, title, im_data=im_data, subscript=subscript)

    def parse_and_log_test_images(self, x, y, y_hat, y_hat_front, y_hat_ym, title, subscript=None, display_count=2):
        im_data = []
        display_count = min(x.shape[0], display_count)
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
                'output_face_Frontal': common.tensor2im(y_hat_front[i]),
                'output_face_YawMirror': common.tensor2im(y_hat_ym[i]),
            }
            im_data.append(cur_im_data)
        distri_utils.fn_on_master_if_distributed(self.opts.distributed_train, self.log_images, title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}_{subscript}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts),
            'global_step': self.global_step,
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg
        if self.opts.dis_lambda > 0:
            save_dict['dis_state_dict'] = self.net_dis.state_dict()
            save_dict['dis_optimizer'] = self.optimizer_dis.state_dict()
        return save_dict

    def configure_scheduler(self, optimizers):
        num_steps = self.opts.max_steps
        warmup_steps = 0  # 28000 * 1.5 #  int(self.opts.max_steps * (1. / 15.))
        min_lr = float(self.opts.learning_rate / 100.)
        warmup_lr = float(self.opts.learning_rate / 1000.)

        lr_schedulers = []
        for optimizer in optimizers:
            lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                t_mul=1.,
                lr_min=min_lr,
                warmup_lr_init=warmup_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
            )
            lr_schedulers.append(lr_scheduler)

        return lr_schedulers

    def train_step_discriminator(self, x, y, y_hat):
        self.net.requires_grad(enc_flag=False, dec_flag=False)
        train_utils.requires_grad(self.net_dis, True)
        self.net.train()
        self.net_dis.train()
        self.optimizer_dis.zero_grad()

        loss_dict = {}
        loss_d_full = 0.
        y.requires_grad_()
        x.requires_grad_()
        d_real = self.net_dis(y, x)
        d_loss_real = train_utils.compute_bce(d_real, 1)
        loss_dict['d_loss_real'] = d_loss_real
        loss_d_full += d_loss_real

        reg = 10. * train_utils.compute_grad2(d_real, self.net_dis.x_in).mean()
        loss_dict['d_loss_reg'] = reg
        loss_d_full += reg

        d_fake = self.net_dis(y_hat.detach(), x)
        d_loss_fake = train_utils.compute_bce(d_fake, 0)
        loss_dict['d_loss_fake'] = d_loss_fake
        loss_d_full += d_loss_fake

        loss_dict['d_loss'] = d_loss_real + d_loss_fake

        if self.opts.distributed_train:
            loss_dict = distri_utils.reduce_dict(loss_dict, average=True)
        loss_dict = {k: v.item() for k, v in loss_dict.items()}

        loss_d_full = loss_d_full * self.opts.dis_lambda
        loss_d_full.backward()
        self.optimizer_dis.step()

        return loss_dict

    def __get_randn_poses(self, batch_size):
        cur_env = self.net.pigan_curriculum
        h_mean_init = cur_env['h_mean']
        v_mean_init = cur_env['v_mean']
        if self.opts.pigan_curriculum_type == 'CelebAMask_HQ':
            h_range = (0.4, 0.55)
            v_range = (0.3, 0.45)
        elif self.opts.pigan_curriculum_type == 'CatMask':
            h_range = (0.3, 0.5)
            v_range = (0.25, 0.4)
        else:
            raise Exception('Range not yet decided for env [%s]' % self.opts.pigan_curriculum_type)

        h_small = np.stack((np.random.uniform(-h_range[1], -h_range[0], size=[batch_size]) + h_mean_init,
                        np.array(v_mean_init).repeat(batch_size)), axis=1)
        h_large = np.stack((np.random.uniform(h_range[0], h_range[1], size=[batch_size]) + h_mean_init,
                            np.array(v_mean_init).repeat(batch_size)), axis=1)
        v_small = np.stack((np.array(h_mean_init).repeat(batch_size),
                            np.random.uniform(-v_range[1], -v_range[0], size=[batch_size]) + v_mean_init), axis=1)
        v_large = np.stack((np.array(h_mean_init).repeat(batch_size),
                            np.random.uniform(v_range[0], v_range[1], size=[batch_size]) + v_mean_init), axis=1)
        select_arr = np.concatenate((h_small, h_large, v_small, v_large), axis=0)
        select_idx = np.random.choice(range(batch_size * 4), batch_size)
        select_poses = select_arr[select_idx, :]

        # match the original structure
        poses = [torch.tensor(select_poses[:, 0]), torch.tensor(select_poses[:, 1])]

        return poses
