from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='celebahq_seg_to_3dface', type=str, help='Type of dataset/experiment to run')
        self.parser.add_argument('--train_paths_conf', default=None, type=str)
        self.parser.add_argument('--test_paths_conf', default=None, type=str)
        self.parser.add_argument('--gpu_ids', default='0', type=str, help="gpu ids, i.e., 0,1")

        self.parser.add_argument('--encoder_type', default='SwinEncoder', type=str, help='Which encoder to use [SwinEncoder]')
        self.parser.add_argument('--discriminator_type', default='patch', type=str, help='Which discriminator to use [patch]')
        self.parser.add_argument('--use_cond_dis', action="store_true", help='Whether to use conditional discriminator or not.')
        self.parser.add_argument('--input_nc', default=16+2, type=int, help='Number of input data channels')
        self.parser.add_argument('--label_nc', default=16, type=int, help='Number of input label channels')
        self.parser.add_argument('--test_output_size', default=128, type=int, help='Output size of generator in test time')
        self.parser.add_argument('--src_input_size', default=224, type=int, help='segmentation mask input size, must be 224 for swin tranformer encoder.')

        self.parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=2, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=1, type=int, help='Number of test/inference dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--learning_rate_dis', default=0.00002, type=float, help='Discriminator optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--scheduler_name', default='none', type=str, help='Which lr scheduler to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true', help='Whether to add average latent vector to generate codes from encoder.')

        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--l1_lambda', default=0, type=float, help='L1 loss multiplier factor')
        self.parser.add_argument('--w_norm_lambda', default=0.005, type=float, help='W-norm loss multiplier factor')
        self.parser.add_argument('--dis_lambda', default=0.08, type=float, help='Discriminator loss multiplier factor for GAN based training')

        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')

        self.parser.add_argument('--max_steps', default=800000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=2500, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=20000, type=int, help='Model checkpoint interval')

        # arguments for pigan encoder
        self.parser.add_argument('--pigan_infer_ray_step', type=int, default=28)
        self.parser.add_argument('--pigan_curriculum_type', type=str, default='CelebAMask_HQ', help="[CelebAMask_HQ | PseudoCats]")
        self.parser.add_argument('--test_pigan_latent_avg', action="store_true", help='test the original average latent code and then stop.')
        self.parser.add_argument('--pigan_steps_conf', type=str, default='configs/pigan_steps/sem2nerf.yaml')
        self.parser.add_argument('--pigan_infer_max_batch', type=int, default=2400000)
        self.parser.add_argument('--pigan_zdim', type=int, default=256)
        self.parser.add_argument('--pigan_train_lvd', action="store_true", help='whether to use lock view dependence during training.')
        self.parser.add_argument('--pigan_train_hs', action="store_true", help='whether to use hierarchical_sample during training.')

        # distributed training parameters
        self.parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        self.parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
        self.parser.add_argument('--local_rank', default=0, type=int,
                                 help='number of distributed processes')
        self.parser.add_argument('--no_distributed', action='store_true', help='Do not use distributed even on slum cluster.')
        self.parser.add_argument('--distributed_train', action='store_true')
        self.parser.add_argument('--use_multi_nodes', action='store_true', help='detail multi nodes using.')

        # patch base training options
        self.parser.add_argument('--ray_scale_anneal', default=0.0019, type=float)
        self.parser.add_argument('--ray_min_scale', default=0.125, type=float)
        self.parser.add_argument('--ray_max_scale', default=1., type=float)
        self.parser.add_argument('--patch_train', action='store_true')
        self.parser.add_argument('--no_ray_random_shift', action='store_true', help='Do not use ray_random_shift, always sample from center region.')

        # data
        self.parser.add_argument('--use_contour', action='store_true')
        self.parser.add_argument('--use_merged_labels', action="store_true", help='Whether to predict pose or not.')

        # validate setting
        self.parser.add_argument('--val_latent_mask', type=str, default=None, help='Comma-separated list of latents to perform style-mixing with')

        # GAN training setting
        self.parser.add_argument('--train_rand_pose_prob', default=-1, type=float)

        # continue training
        self.parser.add_argument('--continue_training', action='store_true', help='load step, optimizer params for continue training setting.')

        # place holder, will be updated in train3d.py according to the pigan_step config file
        self.parser.add_argument('--image_load_size', default=-1, type=int, help='image load size for dataloader.')
        self.parser.add_argument('--image_crop_size', default=-1, type=int, help='image crop size for dataloader.')
        self.parser.add_argument('--output_size', default=-1, type=int, help='Output size of generator')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
