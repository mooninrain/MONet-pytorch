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
from copy import deepcopy
from options.mask_options import MaskOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

import jaclearn.vision.coco.mask_utils as mask_utils

if __name__ == '__main__':
    opt = MaskOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    opt_train = deepcopy(opt)
    opt_train.dataset_type = 'train'
    dataset_train = create_dataset(opt_train)  # create a dataset given opt.dataset_mode and other options
    opt_val = deepcopy(opt)
    opt_val.dataset_type = 'val'
    dataset_val = create_dataset(opt_val)  # create a dataset given opt.dataset_mode and other options
    opt_test = deepcopy(opt)
    opt_test.dataset_type = 'test'
    dataset_test = create_dataset(opt_test)  # create a dataset given opt.dataset_mode and other options

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    with open(os.path.join(opt.dataroot,'scenes','CLEVR_train_scenes.json'),'r') as r:
        data_train_scenes = json.load(r)
    for i, data in enumerate(dataset_train):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        masks = [mask_utils.encode(np.array(visuals['m%d'%i].squeeze().unsqueeze(-1).numpy()>=0,dtype=np.uint8,order='F')) for i in range(11)]

        print(model.get_image_paths())
        print(masks[0])
        break

        # img_path = model.get_image_paths()     # get image paths
        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))