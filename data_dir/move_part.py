import os
import torch
import random
import shutil

ori_dir = '/data/vision/billf/scratch/ruidongwu/data/data_v0'
to_dir = '/data/vision/billf/scratch/ruidongwu/work/decomp/monet/data_dir/partnet'

to_move_dirs = os.listdir(ori_dir)

print('moving pngs...')
for k, _dir_ in enumerate(to_move_dirs):
    print('{:d}/{:d}\r'.format(k,len(to_move_dirs)),end='',flush=True)
    to_move_png = os.path.join(ori_dir,_dir_,'parts_render_after_merging','0.png')
    shutil.copy(to_move_png,os.path.join(to_dir,_dir_+'.png'))