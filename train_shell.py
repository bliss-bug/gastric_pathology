import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--seed', default=2025, type=int)
parser.add_argument('--device', default='cuda:2', type=str)
parser.add_argument('--train_dtfd', default=False, type=bool)
parser.add_argument('--train_gigapath', default=False, type=bool)
args = parser.parse_args()

extractions = ['gigapath', 'uni', 'resnet50']
models = ['abmil', 'dsmil', 'clam_sb', 'clam_mb']
models = ['transmil', 'rrtmil', 'longmil']
models = []

size = {'gigapath': 1536, 'uni': 1024, 'resnet50': 1024}
device = args.device
epochs = args.epochs
seed = args.seed

for model in models:
    for extraction in extractions:
        for i in range(1, 6):
            cmd = f'python train.py --feat_size={size[extraction]} --data_path \'WSI/features/{extraction}_features\' \
                \'WSI/features2/{extraction}_features\' \'WSI/features3/{extraction}_features\' \
                --model=\'{model}\' --fold={i} --device={device} --epochs={epochs} --seed={seed}'
            os.system(cmd)


if args.train_dtfd:
    for extraction in extractions:
        for i in range(1, 6):
            cmd = f'python train_DTFD.py --feat_size={size[extraction]} --data_path \'WSI/features/{extraction}_features\' \
                \'WSI/features2/{extraction}_features\' \'WSI/features3/{extraction}_features\' \
                --fold={i} --device={device} --epochs={epochs} --seed={seed}'
            os.system(cmd)

        
if args.train_gigapath:
    for i in range(1, 6):
        cmd = f'python train_gigapath.py --fold={i} --device={device} --epochs={epochs} --seed={seed}'
        os.system(cmd)
