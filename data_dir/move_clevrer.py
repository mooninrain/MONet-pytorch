import os
import torch
import shutil

ori_dir = '/data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/data/clevrer/ver1.0/frames'
to_dir = '/data/vision/billf/scratch/ruidongwu/work/MONet-pytorch/data_dir/clevrer/images'
to_move_dirs = os.listdir(ori_dir)
to_move_dirs.sort()
torch.manual_seed(123)
perm = torch.randperm(len(to_move_dirs))
to_move_dirs = to_move_dirs[perm]

train_dirs = to_move_dirs[:int(0.8*len(to_move_dirs))]
val_dirs = to_move_dirs[int(0.8*len(to_move_dirs)):int(0.9*len(to_move_dirs))]
test_dirs = to_move_dirs[int(0.9*len(to_move_dirs)):]

if not os.path.exists(to_dir):
    os.makedirs(os.path.join(to_dir,'train'))
    os.makedirs(os.path.join(to_dir,'val'))
    os.makedirs(os.path.join(to_dir,'test'))

print('moving train pngs...')
for k, _dir_ in enumerate(train_dirs):
    print('{:d}/{:d}\r'.format(k,len(train_dirs)),end='',flush=True)
    to_move_pngs = os.listdir(os.path.join(ori_dir,_dir_))
    to_move_pngs = random.sample(to_move_pngs,1)
    for _png_ in to_move_pngs:
        shutil.copy(os.path.join(ori_dir,_dir_,_png_),os.path.join(to_dir,'train',_dir_+'_'+_png_))

print('moving val pngs...')
for k, _dir_ in enumerate(val_dirs):
    print('{:d}/{:d}\r'.format(k,len(val_dirs)),end='',flush=True)
    to_move_pngs = os.listdir(os.path.join(ori_dir,_dir_))
    to_move_pngs = random.sample(to_move_pngs,1)
    for _png_ in to_move_pngs:
        shutil.copy(os.path.join(ori_dir,_dir_,_png_),os.path.join(to_dir,'val',_dir_+'_'+_png_))

print('moving test pngs...')
for k, _dir_ in enumerate(test_dirs):
    print('{:d}/{:d}\r'.format(k,len(test_dirs)),end='',flush=True)
    to_move_pngs = os.listdir(os.path.join(ori_dir,_dir_))
    to_move_pngs = random.sample(to_move_pngs,1)
    for _png_ in to_move_pngs:
        shutil.copy(os.path.join(ori_dir,_dir_,_png_),os.path.join(to_dir,'test',_dir_+'_'+_png_))
