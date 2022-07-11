from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    'celebahq_seg_to_3dface': {
        'transforms': transforms_config.SegToImageTransforms3D,
        'train_source_root': dataset_paths['celebahq_train_segmentation'],
        'train_target_root': dataset_paths['celebahq_train'],
        'test_source_root': dataset_paths['celebahq_test_segmentation'],
        'test_target_root': dataset_paths['celebahq_test'],
    },
    'pseudocats_seg_to_3dface': {
        'transforms': transforms_config.SegToCatTransforms3D,
        'train_source_root': dataset_paths['pseudocats_train_segmentation'],
        'train_target_root': dataset_paths['pseudocats_train'],
        'test_source_root': dataset_paths['pseudocats_test_segmentation'],
        'test_target_root': dataset_paths['pseudocats_test'],
    },
}
