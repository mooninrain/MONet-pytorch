"""This module contains simple helper functions """
from __future__ import print_function
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import os

import cv2
import jaclearn.vision.coco.mask_utils as mask_utils

def crop_and_resize(img,opt)
    img = TF.resized_crop(img, 64, 29, opt.crop_size, opt.crop_size, opt.load_size)
    img = TF.to_tensor(img)
    img = TF.normalize(img, [0.5] * opt.input_nc, [0.5] * opt.input_nc)
    return img

def resize_and_pad(img):