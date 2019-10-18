"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import json
import numpy as np
from copy import deepcopy
from options.mask_options import MaskOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

import cv2
import jaclearn.vision.coco.mask_utils as mask_utils

def mask_preprocess(in_masks,clean=False):
    out_masks = []
    for index, mask in enumerate(in_masks):
        # if index==0:
        #     continue
        mask = cv2.resize(mask, dsize=(480, 320), interpolation=cv2.INTER_CUBIC)
        temp_mask = np.array(mask>0,dtype=np.uint8)
        temp_mask = cv2.blur(temp_mask,(3,3))
        temp_mask = np.array(temp_mask,dtype=np.uint8,order='F')

        # if np.sum(temp_mask)<=20:
        #     continue
        print(np.sum(temp_mask))
        out_masks.append(temp_mask)

    check=input('check')
    return out_masks

def mask_compress(in_masks):
    out_rles = []
    for mask in in_masks:
        temp = mask_utils.encode(mask)
        temp['counts'] = temp['counts'].decode()
        out_rles.append({'mask':temp})
    return out_rles

if __name__ == '__main__':
    opt = MaskOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)

    opt_train = deepcopy(opt)
    opt_train.dataset_type = 'train'
    dataset_train = create_dataset(opt_train)  # create a dataset given opt.dataset_mode and other options
    with open(os.path.join(opt.dataroot,'scenes','CLEVR_train_scenes.json'),'r') as r:
        data_train_scenes = json.load(r)
    for i, data in enumerate(dataset_train):
        if i % 5 == 0:
            print("{:d}/{:d}\r".format(i,len(dataset_train)),end='',flush=True)
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        rles = mask_compress(mask_preprocess([visuals['m%d'%i].squeeze().unsqueeze(-1).cpu().numpy() for i in range(11)]))
        data_train_scenes['scenes'][int(model.get_image_paths()[0][-10:-4])]['objects_detection'] = rles
    with open(os.path.join(opt.results_dir,'scenes_train.json'),'w') as w:
        print('saving scenes_train.json...')
        json.dump(data_train_scenes,w)

    opt_val = deepcopy(opt)
    opt_val.dataset_type = 'val'
    dataset_val = create_dataset(opt_val)  # create a dataset given opt.dataset_mode and other options
    with open(os.path.join(opt.dataroot,'scenes','CLEVR_val_scenes.json'),'r') as r:
        data_val_scenes = json.load(r)
    for i, data in enumerate(dataset_val):
        if i % 5 == 0:
            print("{:d}/{:d}\r".format(i,len(dataset_train)),end='',flush=True)
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        rles = mask_compress(mask_preprocess([visuals['m%d'%i].squeeze().unsqueeze(-1).cpu().numpy() for i in range(11)]))
        data_val_scenes['scenes'][int(model.get_image_paths()[0][-10:-4])]['objects_detection'] = rles
    with open(os.path.join(opt.results_dir,'scenes_val.json'),'w') as w:
        print('saving scenes_val.json...')
        json.dump(data_val_scenes,w)