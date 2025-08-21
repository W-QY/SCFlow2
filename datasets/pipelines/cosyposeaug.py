import random
from mmcv.utils import build_from_cfg
from PIL import ImageEnhance, ImageFilter

from .builder import PIPELINES
from .color_transform import ColorTransform
from ..utils import cv2_to_pil, pil_to_cv2


@PIPELINES.register_module()
class CosyPoseAug(ColorTransform):
    def __init__(self, p=0.8, pipelines=[], patch_level=True, image_keys=['img']):
        super().__init__(patch_level, image_keys)
        self.p = p
        self.pipelines = [
           build_from_cfg(p, PIPELINES) for p in pipelines
        ]

    def augment(self, image):
        if random.random() > self.p:
            return image 
        pil_image = cv2_to_pil(image)
        for p in self.pipelines:
            pil_image = p(pil_image)
        return pil_to_cv2(pil_image)

@PIPELINES.register_module()
class PillowBlur:
    def __init__(self, p=0.4, factor_interval=(1, 3)):
        self.p = p
        self.factor_intervel = factor_interval
    
    def __call__(self, image):
        k = random.randint(*self.factor_intervel)
        image = image.filter(ImageFilter.GaussianBlur(k))
        return image


class PillowRGBAugmentation:
    def __init__(self, pillow_fn, p, factor_interval):
        self._pillow_fn = pillow_fn
        self.p = p
        self.factor_interval = factor_interval

    def __call__(self, image):
        # pil image here, don't check
        if random.random() <= self.p:
            image = self._pillow_fn(image).enhance(factor=random.uniform(*self.factor_interval))
        return image

@PIPELINES.register_module()
class PillowSharpness(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0., 50.)):
        super().__init__(pillow_fn=ImageEnhance.Sharpness,
                         p=p,
                         factor_interval=factor_interval)
@PIPELINES.register_module()
class PillowContrast(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.2, 50.)):
        super().__init__(pillow_fn=ImageEnhance.Contrast,
                         p=p,
                         factor_interval=factor_interval)

@PIPELINES.register_module()
class PillowBrightness(PillowRGBAugmentation):
    def __init__(self, p=0.5, factor_interval=(0.1, 6.0)):
        super().__init__(pillow_fn=ImageEnhance.Brightness,
                         p=p,
                         factor_interval=factor_interval)

@PIPELINES.register_module()
class PillowColor(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.0, 20.0)):
        super().__init__(pillow_fn=ImageEnhance.Color,
                         p=p,
                         factor_interval=factor_interval)
