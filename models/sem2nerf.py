"""
This file defines the core research contribution
"""
from configs.paths_config import model_paths
from models.pigan.op import siren, curriculums
from models.pigan import model as pigan_model
import yaml
from utils.common import filt_ckpt_keys
from utils.train_utils import requires_grad
from models.encoders import swin_encoder
from torch import nn
import torch
import matplotlib
from argparse import Namespace
matplotlib.use('Agg')


def get_keys(d, name):
    return filt_ckpt_keys(d, 'state_dict', name)


class Sem2NeRF(nn.Module):

    def __init__(self, opts):
        super(Sem2NeRF, self).__init__()
        self.set_opts(opts)

        # set encoder
        self.encoder = self.set_encoder()

        # set pigan decoder
        self.pigan_curriculum = getattr(curriculums, opts.pigan_curriculum_type)
        siren_model = getattr(siren, self.pigan_curriculum['model'])
        self.decoder = getattr(pigan_model, self.pigan_curriculum['generator'])(siren_model, opts.pigan_zdim) # .to(self.opts.device)

        # Load weights if needed
        self.load_weights()

    def update_pigan_curriculum(self):
        with open(self.opts.pigan_steps_conf, 'r') as f:
            train_steps_env = yaml.load(f, Loader=yaml.Loader)
        self.pigan_curriculum.update(train_steps_env)

    def set_devices(self, gpu_ids, mode='train'):
        self.enc_device = gpu_ids[0]
        self.dec_device = gpu_ids[-1]

        self.encoder = self.encoder.to(self.enc_device)
        self.decoder = self.decoder.to(self.dec_device)
        self.decoder.set_device(self.dec_device)
        if hasattr(self, 'latent_avg'):
            self.latent_avg = [x.to(self.dec_device) for x in self.latent_avg]

        if self.opts.distributed_train and mode == 'train':
            self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder, device_ids=[self.enc_device], find_unused_parameters=True)
            self.decoder = torch.nn.parallel.DistributedDataParallel(self.decoder, device_ids=[self.dec_device], find_unused_parameters=True)

        self.running_mode = mode

    def set_encoder(self):
        if self.opts.encoder_type == 'SwinEncoder':
            # hard code to match 'swin_tiny_patch4_window7_224.yaml' at the moment
            encoder = swin_encoder.SwinTransformer(
                img_size=224, patch_size=4, in_chans=self.opts.input_nc, num_classes=512*9, embed_dim=96, depths=[2, 2, 6, 2], 
                num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., drop_path_rate=0.2, ape=False, patch_norm=True, use_checkpoint=False
            )
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading Sem2NeRF from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            # Load pretrained weights for SwinEncoder
            if self.opts.encoder_type in ['SwinEncoder']:
                print('Loading encoders weights from swin_tiny_patch4_window7_224!')
                encoder_ckpt = torch.load(model_paths['swin_tiny'], map_location='cpu')['model']
                if self.opts.label_nc != 0:
                    encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if not ('patch_embed' in k or 'head' in k)}
                else:
                    encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if not ('head' in k)}
                self.encoder.load_state_dict(encoder_ckpt, strict=False)
            else:
                raise Exception('Unknown encoder type [%s]' % self.opts.encoder_type)
            
            # load pigan decoder pretrained weights
            print('Loading decoder weights from pretrained!')
            if self.opts.pigan_curriculum_type == "CelebAMask_HQ":
                pigan_model_paths = model_paths['pigan_celeba']
            elif self.opts.pigan_curriculum_type == "CatMask":
                pigan_model_paths = model_paths['pigan_cat']
            else:
                raise Exception("Cannot find environment %s" % self.opts.pigan_curriculum_type)
            ckpt = torch.load(pigan_model_paths, map_location='cpu')
            self.decoder.load_state_dict(ckpt, strict=False)
            if self.opts.start_from_latent_avg:
                self.__load_latent_avg(ckpt, repeat=1)

    def forward(self, x, pose, return_latents=False, iter_num=0):
        # update pigan curriculum
        cur_pigan_env = curriculums.extract_metadata(self.pigan_curriculum, iter_num)
        batch_size = cur_pigan_env['batch_size']
        x = x[:batch_size].to(self.enc_device)
        yaw, pitch = pose
        cur_pigan_env['h_mean'] = yaw[:batch_size].unsqueeze(-1).to(self.dec_device)
        cur_pigan_env['v_mean'] = pitch[:batch_size].unsqueeze(-1).to(self.dec_device)
        cur_pigan_env['lock_view_dependence'] = self.opts.pigan_train_lvd
        cur_pigan_env['hierarchical_sample'] = self.opts.pigan_train_hs

        # map input to latent space
        enc_out_dict = self.encoder(x)
        codes = enc_out_dict['latent_code']
        if self.opts.start_from_latent_avg:
            freq, shift = codes
            codes = (freq.to(self.dec_device) + self.latent_avg[0].repeat(freq.shape[0], 1),
                    shift.to(self.dec_device) + self.latent_avg[1].repeat(shift.shape[0], 1))

        # map latent code to output, need to update pigan decoder accordingly to match patch-scale first
        if self.opts.patch_train:
            cur_pigan_env['opts'] = {
                'patch_train': True,
                'resolution_vol': cur_pigan_env['resolution_vol'],
                'ray_scale_anneal': self.opts.ray_scale_anneal,
                'ray_min_scale': self.opts.ray_min_scale, 'ray_max_scale': self.opts.ray_max_scale,
                'ray_random_scale': True, 
                'ray_random_shift': not self.opts.no_ray_random_shift,
                'iter': iter_num
            }
            # print(cur_pigan_env['resolution_vol'], cur_pigan_env['img_size'])
        else:
            cur_pigan_env['opts'] = {
                'patch_train': False
            }
            cur_pigan_env['img_size'] = cur_pigan_env['resolution_vol']
        cur_pigan_env['opts'] = Namespace(**cur_pigan_env['opts'])
        self.cur_pigan_env = cur_pigan_env
        dec_out_dict = self.decoder(codes, **cur_pigan_env)

        # gather results
        out_dict = {'images': dec_out_dict['pixels']}
        if return_latents:
            out_dict['latents'] = (codes, (self.latent_avg[0].repeat(codes[0].shape[0], 1),
                                        self.latent_avg[1].repeat(codes[1].shape[0], 1)))
        if self.opts.patch_train:
            out_dict['sample_pattern'] = dec_out_dict['sample_pattern']

        return out_dict

    def forward_inference(self, x, pose, latent_mask=None,
                          inject_latent=None, return_latents=False):
        # update pigan curriculum
        cur_pigan_env = curriculums.extract_metadata(self.pigan_curriculum, 0)
        cur_pigan_env['batch_size'] = self.opts.test_batch_size
        cur_pigan_env['num_steps'] = self.opts.pigan_infer_ray_step
        cur_pigan_env['img_size'] = self.opts.test_output_size
        cur_pigan_env['max_batch_size'] = self.opts.pigan_infer_max_batch
        cur_pigan_env['lock_view_dependence'] = True
        cur_pigan_env['hierarchical_sample'] = True
        cur_pigan_env['last_back'] = True
        x = x.to(self.enc_device)

        # map input to latent code
        enc_out_dict = self.encoder(x)
        codes = enc_out_dict['latent_code']
        if self.opts.start_from_latent_avg:
            freq, shift = codes
            codes = (freq.to(self.dec_device) + self.latent_avg[0].repeat(freq.shape[0], 1),
                    shift.to(self.dec_device) + self.latent_avg[1].repeat(shift.shape[0], 1))

        # inject latent space to general multi-modal outputs
        if latent_mask is not None:
            freq, shift = codes
            inject_freq, inject_shift = inject_latent
            for i in latent_mask:
                start_i = i * self.subnet_attri_getter('decoder').siren.hidden_dim
                end_i = (i + 1) * self.subnet_attri_getter('decoder').siren.hidden_dim
                if inject_latent is not None:
                    freq[:, start_i:end_i] = inject_freq[:, start_i:end_i]
                    shift[:, start_i:end_i] = inject_shift[:, start_i:end_i]
                else:
                    freq[:, start_i:end_i] = 0
                    shift[:, start_i:end_i] = 0
            codes = (freq, shift)

        # if pose is not provided, canonical pose will be used as in the default pigan setting
        if pose is not None:
            yaw, pitch = pose
            cur_pigan_env['h_mean'] = yaw.to(self.dec_device).unsqueeze(-1)
            cur_pigan_env['v_mean'] = pitch.to(self.dec_device).unsqueeze(-1)

        # decode latent code to output
        images, _ = self.subnet_attri_getter('decoder').staged_forward(codes, **cur_pigan_env)

        # gather results
        out_dict = {'images': images}
        if return_latents:
            out_dict['latents'] = (codes, (self.latent_avg[0].repeat(codes[0].shape[0], 1),
                                            self.latent_avg[1].repeat(codes[1].shape[0], 1)))
        return out_dict

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = (ckpt['latent_avg'][0], ckpt['latent_avg'][1])
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            z = torch.randn((10000, self.decoder.z_dim))
            with torch.no_grad():
                frequencies, phase_shifts = self.decoder.siren.mapping_network(z)
            avg_frequencies = frequencies.mean(0, keepdim=True)
            avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
            self.latent_avg = (avg_frequencies, avg_phase_shifts)

    def subnet_attri_getter(self, subnet):
        if self.opts.distributed_train and self.running_mode == 'train':
            return getattr(getattr(self, subnet), 'module')
        else:
            return getattr(self, subnet)

    def requires_grad(self, enc_flag, dec_flag=False):
        requires_grad(self.encoder, enc_flag)
        requires_grad(self.decoder, dec_flag)
