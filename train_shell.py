import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--device', default='cuda:2', type=str)
args = parser.parse_args()

extractions = ['gigapath', 'uni', 'resnet50']
#models = ['abmil', 'dsmil', 'clam_sb', 'clam_mb', 'transmil', 'rrtmil']
models = []

size = {'gigapath': 1536, 'uni': 1024, 'resnet50': 1024}
device = args.device
epochs = args.epochs

for model in models:
    for extraction in extractions:
        for i in range(1, 6):
            cmd = f'python train.py --feat_size={size[extraction]} --data_path \'WSI/features/{extraction}_features\' \
                \'WSI/features2/{extraction}_features\' \'WSI/features3/{extraction}_features\' \
                --model=\'{model}\' --fold={i} --device={device} --epochs={epochs}'
            os.system(cmd)

'''
for extraction in extractions:
    for i in range(1, 6):
        cmd = f'python train_DTFD.py --feat_size={size[extraction]} --data_path \'WSI/features/{extraction}_features\' \
            \'WSI/features2/{extraction}_features\' \'WSI/features2/{extraction}_features\' \
            --fold={i} --device={device} --epochs={epochs}'
        os.system(cmd)
'''


for i in range(1, 6):
    cmd = f'python train_gigapath.py --fold={i} --device={device} --epochs={epochs}'
    os.system(cmd)
