"""This module contains simple helper functions """
from __future__ import print_function
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import os

import cv2
import jaclearn.vision.coco.mask_utils as mask_utils

def crop_and_resize(img,i,j,ori_len,resize_len):
    img = TF.resized_crop(img, i, j, ori_len, ori_len, resize_len)
    img = TF.to_tensor(img)
    return img

def resize_and_pad(img,i,j,ori_len,ori_h,ori_w):
    img = TF.resize(img,(ori_len,ori_len))
    img = TF.pad(img,(j,i,ori_w-j-ori_len,ori_h-i-ori_len),fill=0,padding_mode='constant')
    return img