import torch
import torch.nn as nn
from models.discriminators.patch_dis import PatchDiscriminator
from utils.common import filt_ckpt_keys


class Discriminator(nn.Module):
    """docstring for Discriminator."""
    def __init__(self, img_size, opts):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.opts = opts

        input_nc = 3
        if opts.use_cond_dis:
            input_nc = input_nc + self.opts.input_nc
        self.input_nc = input_nc

        self.model = self.set_model()
        self.load_weights()

    def set_model(self):
        # add dis type
        if self.opts.discriminator_type == 'patch':
            model = PatchDiscriminator(self.img_size, input_nc=self.input_nc)
        else:
            raise Exception("Unknow discriminator_type [%s]" % self.opts.discriminator_type)
        return model
    
    def set_devices(self, gpu_ids):
        self.device = gpu_ids[0]
        self.model.to(self.device)
        if self.opts.distributed_train:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device], find_unused_parameters=True)

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            updated_ckpt = filt_ckpt_keys(ckpt, 'dis_state_dict', 'model')
            self.model.load_state_dict(updated_ckpt, strict=True)
            print('Loaded discriminator from checkpoint: {}'.format(self.opts.checkpoint_path))

    def forward(self, x, label=None):
        if self.opts.use_cond_dis and label is not None:
            self.x_in = torch.cat([x, label], dim=1)
        else:
            self.x_in = x

        return self.model(self.x_in)
