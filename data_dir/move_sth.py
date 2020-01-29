import os
import random
import shutil

random.seed(123)

ori_dir = '/data/vision/billf/scratch/ruidongwu/work/decomp/monet/data_dir/sth/20bn-something-something-v1'
to_dir = '/data/vision/billf/scratch/ruidongwu/work/decomp/monet/data_dir/sthsth/images'
to_move_dirs = os.listdir(ori_dir)

print('moving train pngs...')
for k, _dir_ in enumerate(to_move_dirs):
    print('{:d}/{:d}\r'.format(k,len(to_move_dirs)),end='',flush=True)
    to_move_jpgs = os.listdir(os.path.join(ori_dir,_dir_))
    to_move_jpgs = random.sample(to_move_jpgs,1)
    for _jpg_ in to_move_jpgs:
        shutil.copy(os.path.join(ori_dir,_dir_,_jpg_),os.path.join(to_dir,_dir_+'_'+_jpg_))