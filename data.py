import os, glob
import pickle

from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
    

class PatchDataset(Dataset):
    def __init__(self, transform, slide_dir='WSI/GPI/single', surfix='jpeg'):
        SlideNames = os.listdir(slide_dir)
        SlideNames = [sst for sst in SlideNames if os.path.isdir(os.path.join(slide_dir, sst))]

        self.patch_dirs = []
        for tslideName in SlideNames:
            tpatch_paths = glob.glob(os.path.join(slide_dir, tslideName, '*.'+surfix))
            self.patch_dirs.extend(tpatch_paths)

        self.transform = transform

    def __getitem__(self, idx):
        img_dir = self.patch_dirs[idx]
        image = Image.open(img_dir)
        image = self.transform(image)

        slide_name = img_dir.split('/')[-2]
        img_name = os.path.basename(img_dir).split('.')[0]

        return image, slide_name, img_name
        

    def __len__(self):
        return len(self.patch_dirs)



class PathologyDataset(Dataset):
    def __init__(self, feats_path, labels):
        self.feats_path = feats_path
        self.labels = labels

    def __getitem__(self, idx):
        with open(self.feats_path[idx], 'rb') as f:
            data = pickle.load(f)
        feat = np.array([d['feature'] for d in data])
        pos = np.array([[int(d['file_name'].split('_')[0]), int(d['file_name'].split('_')[1])] for d in data])

        id = self.feats_path[idx].split('/')[-1].split('.')[0].split('-')[0].split('H')[0]
        label = self.labels[id]

        return feat, pos, label, id

    def __len__(self):
        return len(self.feats_path)


if __name__ == '__main__':
    d = PathologyDataset(['WSI/features/gigapath_features/201753232.pkl', 
                          'WSI/features/gigapath_features/F201801552.pkl'], 
                          {'201753232': 0, 'F201801552': 1})

    loader = DataLoader(dataset=d)

    for c in loader:
        print(c[1].shape)