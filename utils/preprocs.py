import cv2
import imgaug
import numpy as np
from PIL import ImageOps


def clahe(img):
    cv2.setNumThreads(1)
    eq = cv2.createCLAHE(40., (8, 8))
    return np.array([eq.apply(img[:, :, i]) for i in range(img.shape[-1])], dtype=np.uint8).transpose(1, 2, 0)
    # return equalize_clahe(img)


_aug_combo = imgaug.augmenters.Sequential([
    imgaug.augmenters.Sometimes(0.3, imgaug.augmenters.OneOf([
        imgaug.augmenters.pillike.EnhanceBrightness(),
        imgaug.augmenters.pillike.EnhanceContrast()
    ])),
    imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.pillike.EnhanceSharpness()),
    imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.AdditiveGaussianNoise(scale=(0, 0.03 * 255))),
    imgaug.augmenters.Sometimes(0.3, imgaug.augmenters.OneOf([
        imgaug.augmenters.MotionBlur(k=(3, 9), angle=[-45, 45]),
        imgaug.augmenters.GaussianBlur(sigma=(0, 1.3))
    ]))
])


def aug_combo(img):
    return _aug_combo(image=img)


def padding(img, expected_size=(320, 320)):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size=(256, 256)):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)
