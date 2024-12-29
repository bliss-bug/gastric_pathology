import os
import math
import argparse
import pickle
import openslide
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from model.lbmil import LearnableBiasMIL


def compute_attention(milnet: LearnableBiasMIL, feat_path):
    with open(feat_path, 'rb') as f:
        data = pickle.load(f)
    feat = np.array([d['feature'] for d in data])
    pos = np.array([[int(d['file_name'].split('_')[0]), int(d['file_name'].split('_')[1])] for d in data])
    
    milnet.eval()
    with torch.no_grad():
        _, _, Y_prob, attn = milnet(torch.tensor(np.concatenate([feat, pos], axis=1), dtype=torch.float).cuda(0))
        print(Y_prob)
        weight = torch.sum(attn.squeeze(), (0, 1)).cpu().numpy()

    # Normalize weights
    weight = (weight - weight.min()) / (weight.max() - weight.min())

    sorted_idx = np.argsort(-weight)
    print(pos[sorted_idx[:10]])

    return pos, weight



def compute_gradcam(model: LearnableBiasMIL, feat_path, target_class=1):
    with open(feat_path, 'rb') as f:
        data = pickle.load(f)
    feat = np.array([d['feature'] for d in data])
    pos = np.array([[int(d['file_name'].split('_')[0]), int(d['file_name'].split('_')[1])] for d in data])
    x = torch.tensor(np.concatenate([feat, pos], axis=1), dtype=torch.float).cuda(0)

    model.eval()
    logits, _, Y_prob, attn = model(x)
    print(Y_prob)

    # Get target class score
    score = logits[:, target_class]  # Shape: [1]

    # Backpropagate to get gradients w.r.t. h
    model.zero_grad()
    score.backward(retain_graph=True)

    # Access gradients and activations
    gradients = model.saved_h.grad.squeeze(0)  # Shape: [N, feat_size]
    activations = model.saved_h.squeeze(0)    # Shape: [N, feat_size]

    # Compute channel-wise weights (global average pooling)
    weights = gradients.mean(dim=0)  # Shape: [feat_size]

    # Weighted sum of activations
    cam = torch.matmul(activations, weights)  # Shape: [N]

    # ReLU and normalize
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize to [0, 1]

    cam = cam.detach().cpu().numpy()
    sorted_idx = np.argsort(-cam)
    print(pos[sorted_idx[:10]])

    return pos, cam



def generate_heatmap(ndpi_file, output_file, scale_factor, pos, weight):
    # 打开 ndpi 文件
    slide = openslide.OpenSlide(ndpi_file)
    
    # 使用较低分辨率层级读取图像
    num_levels = slide.level_count
    MAG_BASE = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    selected_level = min(num_levels - 1, int(math.log2(float(MAG_BASE) / scale_factor)))
    level_dimensions = slide.level_dimensions[selected_level]
    
    print(f"选择的层级为 Level {selected_level}，分辨率为 {level_dimensions}")
    
    # 读取缩放后的图像
    region = slide.read_region((0, 0), selected_level, level_dimensions).convert("RGB")
    img = np.array(region)

    # 创建权重矩阵
    size = int(256 / (20 / scale_factor))
    height, width, _ = img.shape
    weights = np.zeros((height, width))
    for (x, y), w in zip(pos, weight):
        weights[y*size:y*size+size, x*size:x*size+size] = w
    
    # 创建热力图
    cmap = plt.get_cmap('jet')  # 使用 Jet 色图
    heatmap = cmap(weights)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)  # 转为 RGB 格式

    # 将热力图叠加到原始图像
    overlay = (0.6 * img + 0.4 * heatmap).astype(np.uint8)

    # 保存叠加结果
    overlay_image = Image.fromarray(overlay)
    overlay_image.save(output_file, format="JPEG")
    print(f"热力图已保存为 {output_file}")



def main(args):
    ndpi_file = args.ndpi_file
    scale_factor = args.scale_factor
    output_jpg = f"{args.heatmap_type}/{args.ndpi_file.split('/')[-1].split('.')[0]}.jpg"

    milnet = LearnableBiasMIL(input_size=args.feat_size, n_classes=2).cuda(0)
    milnet.load_state_dict(torch.load(args.checkpoint))

    os.makedirs(args.heatmap_type, exist_ok=True)
    if args.heatmap_type == "attentionmap":
        pos, weight = compute_attention(milnet, args.feat_path)
    elif args.heatmap_type == "gradcam":
        pos, weight = compute_gradcam(milnet, args.feat_path)

    generate_heatmap(ndpi_file, output_jpg, scale_factor, pos, weight)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ndpi_file', default="WSI/GPI/S202222247.ndpi", type=str)
    parser.add_argument('--feat_path', default="WSI/features/gigapath_features/S202222247.pkl",type=str)
    parser.add_argument('--heatmap_type', default="attentionmap", type=str)
    parser.add_argument('--checkpoint', default="best_checkpoints/gigapath_lbmil_fold2.pth", type=str)
    parser.add_argument('--scale_factor', default=0.625, type=float)
    parser.add_argument('--feat_size', default=1536, type=int)

    args = parser.parse_args()
    main(args)