import os, argparse
from collections import defaultdict
import pickle
from tqdm import tqdm
from PIL import Image

import timm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data import PatchDataset
from model.Resnet import Resnet50


def compute_feats(model, dataloader, device, save_path):
    model.eval()
    feats = defaultdict(list)
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for images, slide_names, img_names in tqdm(dataloader):
            images = images.to(device)
            fs = model(images)
            fs = fs.cpu().numpy()

            for i in range(fs.shape[0]):
                feats[slide_names[i]].append({'feature': fs[i], 'file_name': img_names[i]})

    for slide, info in feats.items():
        slide_save_path = os.path.join(save_path, slide+'.pkl')
        with open(slide_save_path, 'wb') as f:
            pickle.dump(info, f)



def main(args):
    device = torch.device(args.device) if torch.cuda.is_available() else 'cpu'

    if args.model == 'uni':
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, 
            init_values=1e-5, num_classes=0, dynamic_img_size=True
            ).to(device)
        model.load_state_dict(torch.load("checkpoints/uni.bin"), strict=True)
        
    elif args.model == 'resnet50':
        model = Resnet50().to(device)

    elif args.model == 'gigapath':
        model = torch.load("checkpoints/tile_encoder.pt").to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dataset = PatchDataset(transform=transform, slide_dir=args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    compute_feats(model, dataloader, device, os.path.join(args.save_path, args.model+'_features'))

    '''
    image = Image.open("WSI/GPI/single/201408795/2_86.jpeg")
    image = transform(image).unsqueeze(dim=0).to(device)
    with torch.inference_mode():
        feature_emb = model(image)
        print(feature_emb.shape)
    '''
    



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='gigapath', type=str)
    parser.add_argument('--device', default='cuda:2', type=str)
    parser.add_argument('--data_path', default='WSI/GPI/single', type=str)
    parser.add_argument('--save_path', default='WSI/features', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=256, type=int)

    args = parser.parse_args()

    main(args)