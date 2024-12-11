import argparse
import pickle
import openslide
from PIL import Image
import matplotlib.pyplot as plt

import torch
import numpy as np
from model import lbmil


def generate_weight(milnet, feat_path):
    with open(feat_path, 'rb') as f:
        data = pickle.load(f)
    feat = np.array([d['feature'] for d in data])
    pos = np.array([[int(d['file_name'].split('_')[0]), int(d['file_name'].split('_')[1])] for d in data])
    
    milnet.eval()
    with torch.no_grad():
        _, _, Y_prob, attn = milnet(torch.tensor(np.concatenate([feat, pos], axis=1), dtype=torch.float).cuda(0))
        print(Y_prob)
        weight = torch.sum(attn.squeeze(), (0, 1)).cpu().numpy()

    sorted_idx = np.argsort(-weight)
    print(pos[sorted_idx[:20]])
    return pos, weight



def generate_heatmap(ndpi_file, output_file, scale_factor, pos, weight):
    # 打开 ndpi 文件
    slide = openslide.OpenSlide(ndpi_file)
    
    # 使用较低分辨率层级读取图像
    num_levels = slide.level_count
    selected_level = min(num_levels - 1, int(slide.level_downsamples.index(min(slide.level_downsamples, key=lambda x: abs(x - scale_factor)))))
    level_dimensions = slide.level_dimensions[selected_level]
    
    print(f"选择的层级为 Level {selected_level}，分辨率为 {level_dimensions}")
    
    # 读取缩放后的图像
    region = slide.read_region((0, 0), selected_level, level_dimensions).convert("RGB")
    img = np.array(region)

    # 创建权重矩阵（例如，简单用 y 坐标生成梯度权重）
    height, width, _ = img.shape
    weights = np.zeros((height, width))
    for (x, y), w in zip(pos, weight):
        weights[y*2:y*2+2, x*2:x*2+2] = w - weight.min()

    # 归一化权重到 [0, 1]
    weights_normalized = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    
    # 创建热力图
    cmap = plt.get_cmap('jet')  # 使用 Jet 色图
    heatmap = cmap(weights_normalized)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)  # 转为 RGB 格式

    # 将热力图叠加到原始图像
    overlay = (0.6 * img + 0.4 * heatmap).astype(np.uint8)

    # 保存叠加结果
    overlay_image = Image.fromarray(overlay)
    overlay_image.save(output_file, format="JPEG")
    print(f"热力图已保存为 {output_file}")



def main(args):
    ndpi_file = args.ndpi_file
    output_jpg = args.output_jpg
    scale_factor = args.scale_factor

    milnet = lbmil.LearnableBiasMIL(input_size=args.feat_size, n_classes=2).cuda(0)
    milnet.load_state_dict(torch.load(args.checkpoint))

    pos, weight = generate_weight(milnet, args.feat_path)

    generate_heatmap(ndpi_file, output_jpg, scale_factor, pos, weight)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ndpi_file', default="WSI/GPI/S202228530.ndpi", type=str)
    parser.add_argument('--feat_path', default="WSI/features/uni_features/S202228530.pkl",type=str)
    parser.add_argument('--output_jpg', default="heatmap_outcome/S202228530.jpg", type=str)
    parser.add_argument('--checkpoint', default="checkpoints/uni_lbmil.pth", type=str)
    parser.add_argument('--scale_factor', default=128, type=int)
    parser.add_argument('--feat_size', default=1024, type=int)

    args = parser.parse_args()
    main(args)