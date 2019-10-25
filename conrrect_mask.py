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

import torchvision.transforms.functional as TF
import cv2
import jaclearn.vision.coco.mask_utils as mask_utils

def mask_preprocess(in_masks,clean=False):
    out_masks = []
    for index, mask in enumerate(in_masks):
        temp_mask = np.array(mask>0,dtype=np.uint8)
        temp_mask = cv2.blur(temp_mask,(3,3))

        # temp_mask = TF.resize(temp_mask,(192,192))
        # temp_mask = TF.pad(temp_mask,(29,64,480-29-192,320-64-192))
        temp_mask = cv2.resize(temp_mask,(192,192))
        temp_mask = cv2.copyMakeBorder(temp_mask,64,320-64-192,29,480-29-192,cv2.BORDER_CONSTANT,value=0)

        temp_mask = np.array(temp_mask,dtype=np.uint8,order='F')

        if np.sum(temp_mask)<=1000:
            continue
        out_masks.append(temp_mask)

    if len(out_masks)>1:
        out_masks = out_masks[1:]
    return out_masks

def mask_compress(in_masks):
    out_rles = []
    for mask in in_masks:
        temp = mask_utils.encode(mask)
        temp['counts'] = temp['counts'].decode()
        out_rles.append({'mask':temp})
    return out_rles

# def generate_mask(dataset,model,from_file,to_file):
#     with open(os.path.join(from_file),'r') as r:
#         data_scenes = json.load(r)
#     for i, data in enumerate(dataset):
#         if i % 5 == 0:
#             print("{:d}/{:d}\r".format(i,len(dataset)),end='',flush=True)
#         model.set_input(data)  # unpack data from data loader
#         model.test()           # run inference
#         visuals = model.get_current_visuals()  # get image results
#         rles = mask_compress(mask_preprocess([visuals['m%d'%i].squeeze().unsqueeze(-1).cpu().numpy() for i in range(11)]))
#         data_scenes['scenes'][int(model.get_image_paths()[0][-10:-4])]['objects_detection'] = rles
#     with open(os.path.join(opt.results_dir,'scenes_train.json'),'w') as w:
#         print('saving scenes_train.json...')
#         json.dump(data_scenes,w)



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
    with open('/data/vision/billf/scratch/ruidongwu/work/nscls/NSCL1/data_dir/clevr_monet_mask_ori/train/scenes.json','r') as r:
        data_train_scenes_reference = json.load(r)

    for i, data in enumerate(dataset_train):
        if i % 5 == 0:
            print("{:d}/{:d}\r".format(i,len(dataset_train)),end='',flush=True)
        model.set_input(data)  # unpack data from data loader

        if len(data_train_scenes_reference['scenes'][int(model.get_image_paths()[0][-10:-4])]['objects_detection']) == 0:
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            rles = mask_compress(mask_preprocess([visuals['m%d'%i].squeeze().unsqueeze(-1).cpu().numpy() for i in range(11)]))
            data_train_scenes_reference['scenes'][int(model.get_image_paths()[0][-10:-4])]['objects_detection'] = rles
    with open('/data/vision/billf/scratch/ruidongwu/work/nscls/NSCL1/data_dir/clevr_monet_mask_ori/train/scenes.json','w') as w:
        print('saving scenes_train.json...')
        json.dump(data_train_scenes_reference,w)

