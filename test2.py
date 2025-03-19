import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from data import PathologyDataset
from model import abmil, clam, dsmil, transmil, rrt, longmil, lbmil
from utils import load_test_data


def test(dataloader, milnet, criterion, device, model='lbmil'):
    milnet.eval()
    true_dict, score_dict = defaultdict(int), defaultdict(float)
    with torch.no_grad():
        for feats, poses, labels, id in tqdm(dataloader):
            feats, poses, labels = feats.squeeze().to(device), poses.squeeze().to(device), labels.long().to(device)
            if model == 'lbmil':
                x = torch.cat([feats, poses], dim=1)
                bag_prediction, Y_hat, Y_prob, attention = milnet(x)
                true_dict[id[0]] = labels.item()
                score_dict[id[0]] = max(score_dict[id[0]], torch.squeeze(Y_prob)[1].item())

    y_true, y_pred, y_score = [], [], []
    for key in true_dict.keys():
        y_true.append(true_dict[key])
        y_pred.append(int(score_dict[key] > 0.5))
        y_score.append(score_dict[key])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    specificity = sum((y_true == 0) & (y_pred == 0)) / sum(y_true == 0)
    auc = roc_auc_score(y_true, y_score)

    for k, v in score_dict.items():
        print(k, v)

    return acc, precision, recall, f1, specificity, auc



def main(args):
    device = torch.device(args.device) if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()

    test_path, labels = load_test_data(args.data_path, args.label_path)
    testset = PathologyDataset(test_path, labels)
    testloader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.model == 'abmil':
        milnet = abmil.Attention(in_size=args.feat_size, out_size=args.num_classes).to(device)
    elif args.model == 'clam_sb':
        milnet = clam.CLAM_SB(dropout=0.25, n_classes=args.num_classes, embed_dim=args.feat_size, subtyping=True).to(device)
    elif args.model == 'clam_mb':
        milnet = clam.CLAM_MB(dropout=0.25, n_classes=args.num_classes, embed_dim=args.feat_size, subtyping=True).to(device)
    elif args.model == 'dsmil':
        i_classifier = dsmil.FCLayer(in_size=args.feat_size, out_size=args.num_classes).to(device)
        b_classifier = dsmil.BClassifier(input_size=args.feat_size, output_class=args.num_classes).to(device)
        milnet = dsmil.MILNet(i_classifier, b_classifier).to(device)
    elif args.model == 'transmil':
        milnet = transmil.TransMIL(input_size=args.feat_size, n_classes=args.num_classes).to(device)
    elif args.model == 'rrtmil':
        milnet = rrt.RRTMIL(input_dim=args.feat_size, n_classes=args.num_classes).to(device)
    elif args.model == 'longmil':
        milnet = longmil.LongMIL(n_classes=args.num_classes, input_size=args.feat_size).to(device)
    elif args.model == 'lbmil':
        milnet = lbmil.LearnableBiasMIL(input_size=args.feat_size, n_classes=args.num_classes).to(device)

    milnet.load_state_dict(torch.load(args.checkpoint))
    extraction = args.data_path.split('/')[-1].split('_')[0] if isinstance(args.data_path, str) else args.data_path[0].split('/')[-1].split('_')[0]
    test_acc, test_precision, test_recall, test_f1, test_specificity, test_auc = test(testloader, milnet, criterion, device, args.model)

    print('extraction = {}, model = {}'.format(extraction, args.model))
    print('test: acc = {:.4f} | precision = {:.4f} | recall = {:.4f} | f1 = {:.4f} | specificity = {:.4f} | auc = {:.4f}\n'.format(test_acc, test_precision, test_recall, test_f1, test_specificity, test_auc))



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--feat_size', default=1536, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    
    parser.add_argument('--model', default='lbmil', type=str)
    parser.add_argument('--data_path', default=['WSI/features_in_test/gigapath_features', 'WSI/features_in_test_single/gigapath_features'], type=str)
    parser.add_argument('--label_path', default='labels/all_labels.xlsx', type=str)
    parser.add_argument('--checkpoint', default='work_dirs/gigapath_lbmil/20250312_002548/gigapath_lbmil.pth', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)

    args = parser.parse_args()
    main(args)