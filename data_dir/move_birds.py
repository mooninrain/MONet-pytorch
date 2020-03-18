import os
import shutil

ori_dir = '/data/vision/billf/scratch/ruidongwu/work/decomp/monet/data_dir/birds/CUB_200_2011/images'
to_dir = '/data/vision/billf/scratch/ruidongwu/work/decomp/monet/data_dir/birds_images'

dir_list = os.listdir(ori_dir)
for _dir_ in dir_list:
    temp_path = os.path.join(ori_dir,_dir_)
    dir_list2 = os.listdir(temp_path)
    for _dir_2 in dir_list2:
        temp_path2 = os.path.join(temp_path, _dir_2)
        to_path = os.path.join(to_dir,_dir_2)
        shutil.copy(temp_path2,to_path)