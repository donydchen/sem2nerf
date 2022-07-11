from abc import abstractmethod
import torchvision.transforms as transforms
from datasets import augmentations
from PIL import Image


class TransformsConfig(object):

    def __init__(self, opts):
        self.opts = opts

    @abstractmethod
    def get_transforms(self):
        pass


class SegToImageTransforms3D(TransformsConfig):

    def __init__(self, opts):
        super(SegToImageTransforms3D, self).__init__(opts)

    def get_transforms(self, output_size, src_input_size):
        load_size = self.opts.image_load_size
        crop_size = self.opts.image_crop_size

        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((load_size, load_size)),
                transforms.CenterCrop(crop_size),
                transforms.Resize((output_size, output_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': transforms.Compose([
                transforms.Resize((load_size, load_size), interpolation=Image.NEAREST),
                transforms.CenterCrop(crop_size),
                transforms.Resize((src_input_size, src_input_size), interpolation=Image.NEAREST),
                augmentations.ToOneHot(self.opts.label_nc),
                transforms.ToTensor()]),
            'transform_test': transforms.Compose([
                transforms.Resize((load_size, load_size)),
                transforms.CenterCrop(crop_size),
                transforms.Resize((output_size, output_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((load_size, load_size), interpolation=Image.NEAREST),
                transforms.CenterCrop(crop_size),
                transforms.Resize((src_input_size, src_input_size), interpolation=Image.NEAREST),
                augmentations.ToOneHot(self.opts.label_nc),
                transforms.ToTensor()])
        }
        return transforms_dict


class SegToCatTransforms3D(TransformsConfig):

    def __init__(self, opts):
        super(SegToCatTransforms3D, self).__init__(opts)

    def get_transforms(self, output_size, src_input_size):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((output_size, output_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': transforms.Compose([
                transforms.Resize((src_input_size, src_input_size), interpolation=Image.NEAREST),
                augmentations.ToOneHot(self.opts.label_nc),
                transforms.ToTensor()]),
            'transform_test': transforms.Compose([
                transforms.Resize((output_size, output_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((src_input_size, src_input_size), interpolation=Image.NEAREST),
                augmentations.ToOneHot(self.opts.label_nc),
                transforms.ToTensor()])
        }
        return transforms_dict
