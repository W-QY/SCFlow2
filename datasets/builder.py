from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import ConcatDataset
from copy import deepcopy

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
DATASETS.register_module(name='ConcatDataset', module=ConcatDataset)

def build_dataset(cfg):
    if 'multisourcesample' in cfg:
        cfg_ = deepcopy(cfg)
        cfg_.pop('multisourcesample')
    else:
        cfg_ = cfg
    if cfg_.type == 'ConcatDataset':
        cfg_.datasets = [build_from_cfg(d, DATASETS) for d in cfg_.datasets]
        return build_from_cfg(cfg_, DATASETS)
    else:
        return build_from_cfg(cfg_, DATASETS)
