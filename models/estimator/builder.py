from mmcv.utils import Registry, build_from_cfg


ESTIMATORS = Registry('estimator')


def build_estimator(cfg):
    return build_from_cfg(cfg, ESTIMATORS)