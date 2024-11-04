import os

extractions = ['gigapath', 'uni', 'resnet50']
#extractions = ['resnet50']
#models = ['abmil', 'dsmil', 'transmil', 'rrtmil']
models = ['lbmil']
size = {'gigapath': 1536, 'uni': 1024, 'resnet50': 1024}

for model in models:
    for extraction in extractions:
        for i in range(1, 6):
            cmd = f'python train.py --feat_size={size[extraction]} --data_path \'WSI/features/{extraction}_features\' \
                \'WSI/features2/{extraction}_features\' \'WSI/features3/{extraction}_features\' \
                \'WSI/features4/{extraction}_features\' --model=\'{model}\' --fold={i}'
            os.system(cmd)

'''
for extraction in extractions:
    for i in range(1, 6):
        cmd = f'python train_DTFD.py --feat_size={size[extraction]} --data_path \'WSI/features/{extraction}_features\' \
            \'WSI/features2/{extraction}_features\' \'WSI/features3/{extraction}_features\' \
            \'WSI/features4/{extraction}_features\' --fold={i}'
        os.system(cmd)
'''

'''
for i in range(1, 6):
    cmd = f'python train_gigapath.py --fold={i}'
    os.system(cmd)
'''