import os
import shutil

ori_dir = '/data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/data/clevrer/ver1.0/frames'
to_dir = '/data/vision/billf/scratch/ruidongwu/work/MONet-pytorch/data_dir/clevrer'
to_move_dirs = os.listdir(ori_dir)
to_move_dirs.sort()

if not os.path.exists(to_dir):
    os.makedirs(to_dir)

for k, _dir_ in enumerate(to_move_dirs):
    to_move_pngs = os.listdir(os.path.join(ori_dir,_dir_))
    to_move_pngs.sort()
    for _png_ in to_move_pngs:
        shutil.copy(os.path.join(ori_dir,_dir_,_png_),os.path.join(to_dir,_dir_+'_'+_png_))
