from argparse import ArgumentParser


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # arguments for inference script
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path', default='pretrained_models/sem2nerf_celebahq_pretrained.pt', type=str, help='Path to pSp model checkpoint')
        self.parser.add_argument('--data_path', type=str, default='data/CelebAMask-HQ/mask_samples', help='Path to directory of images to evaluate')
        self.parser.add_argument('--couple_outputs', action='store_true', help='Whether to also save inputs + outputs side-by-side')
        self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs to 256x256 or keep at 1024x1024')
        self.parser.add_argument('--image_load_size', default=320, type=int, help='image load size for dataloader.')
        self.parser.add_argument('--image_crop_size', default=256, type=int, help='image crop size for dataloader.')
        self.parser.add_argument('--test_output_size', default=256, type=int, help='Output size of generator in test time')
        self.parser.add_argument('--src_input_size', default=224, type=int, help='segmentation mask input size, must be 224 for swin tranformer encoder.')
        self.parser.add_argument('--use_original_pose', action='store_true', help='Whether to inverse the image based on the original pose.')
        self.parser.add_argument('--render_videos', action='store_true', help='Whether to render video output.')
        self.parser.add_argument('--trajectory', type=str, default='front')
        self.parser.add_argument('--num_frames', type=int, default=36)
        self.parser.add_argument('--use_shuffle', action='store_true', help='Whether to shuffle the input data')

        self.parser.add_argument('--infer_paths_conf', default=None, type=str)
        self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=1, type=int, help='Number of test/inference dataloader workers')

        # arguments for style-mixing script
        self.parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')
        self.parser.add_argument('--n_outputs_to_generate', type=int, default=5, help='Number of outputs to generate per input image.')
        self.parser.add_argument('--mix_alpha', type=float, default=None, help='Alpha value for style-mixing')
        self.parser.add_argument('--latent_mask', type=str, default='8', help='Comma-separated list of latents to perform style-mixing with')
        self.parser.add_argument('--inject_code_seed', type=int, default=0, help='Number of outputs to generate per input image.')
        self.parser.add_argument('--mask_overlay_weight', type=float, default=0., help='overlay mask on generated output.')
        
        # arguments for super-resolution
        self.parser.add_argument('--resize_factors', type=str, default=None,
                                 help='Downsampling factor for super-res (should be a single value for inference).')

        # arguments for pigan decoder
        self.parser.add_argument('--pigan_infer_ray_step', type=int, default=24)
        self.parser.add_argument('--pigan_curriculum_type', type=str, default='CelebAMask_HQ', help="[CelebAMask_HQ | PseudoCats]")
        self.parser.add_argument('--test_pigan_latent_avg', action="store_true", help='test the original average latent code and then stop.')
        self.parser.add_argument('--pigan_steps_conf', type=str, default='configs/pigan_steps/mask2face_3090.yaml')
        self.parser.add_argument('--pigan_infer_max_batch', type=int, default=50000)

        self.parser.add_argument('--latent_codes_path', type=str, default=None, help='latent_codes_path')
        self.parser.add_argument('--distri_without_z', action='store_true', help='Whether to shuffle the input data')

        self.parser.add_argument('--encoder_not_zplus', action="store_true", help='Whether to use z plus encoding or simply single z space.')

        # paper drawing
        self.parser.add_argument('--render_mode', type=str, default='h', help='h, v, fov')
        self.parser.add_argument('--inject_code_seed2', type=int, default=256, help='Number of outputs to generate per input image.')
        self.parser.add_argument('--style_mix_seq_num', type=int, default=7, help='sequence number for style mixing.')
        self.parser.add_argument('--video_rev_end', action="store_true", help='Whether to use reverse ending.')


        # legacy, to back support testing on old codes
        self.parser.add_argument('--predict_pose', action="store_true", help='Whether to predict pose or not.')
        self.parser.add_argument('--gpu_ids', default='0', type=str, help="gpu ids, i.e., 0,1")
        self.parser.add_argument('--encoder_n_styles', type=int, default=9, help='If it is less than 9, the rest will be set to 0.')
        self.parser.add_argument('--encoder_reverse_codes', action="store_true", help='whether to reverse the latent codes.')
        self.parser.add_argument('--use_merged_labels', action="store_true", help='Whether to predict pose or not.')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
