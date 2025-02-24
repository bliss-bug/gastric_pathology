import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import random
import time
import argparse
from math import inf
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from data import PathologyDataset
from model import abmil, clam, dsmil, transmil, rrt, longmil, lbmil
from utils import load_trainval_data


def train(dataloader, milnet, criterion, optimizer, device, model='lbmil'):
    milnet.train()
    losses, num = 0, 0

    for feats, poses, labels in tqdm(dataloader):
        optimizer.zero_grad()
        feats, poses, labels = feats.squeeze().to(device), poses.squeeze().to(device), labels.long().to(device) # [N, C], [N, 2],[1]
        if model == 'abmil':
            bag_prediction, _, attention = milnet(feats)
            loss = criterion(bag_prediction, labels)
        elif model in ['clam_sb', 'clam_mb']:
            bag_prediction, Y_prob, Y_hat, _, instance_dict = milnet(feats, labels, instance_eval=True)
            bag_loss = criterion(bag_prediction, labels)
            instance_loss = instance_dict['instance_loss']
            loss = 0.7*bag_loss + 0.3*instance_loss
        elif model == 'dsmil':
            ins_prediction, bag_prediction, attention, atten_B = milnet(feats)
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_loss = criterion(bag_prediction, labels)
            max_loss = criterion(max_prediction.unsqueeze(0), labels)
            loss = 0.5*bag_loss + 0.5*max_loss
        elif model == 'transmil':
            output = milnet(feats)
            bag_prediction, bag_feature = output['logits'], output['Bag_feature']
            loss = criterion(bag_prediction, labels)
        elif model == 'rrtmil':
            bag_prediction = milnet(feats)
            loss = criterion(bag_prediction, labels)
        elif model == 'longmil':
            x = torch.cat([feats, poses], dim=1)
            bag_prediction, _, _, _ = milnet(x)
            loss = criterion(bag_prediction, labels)
        elif model == 'lbmil':
            x = torch.cat([feats, poses], dim=1)
            bag_prediction, _, _, _ = milnet(x)
            loss = criterion(bag_prediction, labels)

        loss.backward()
        optimizer.step()

        losses += loss.item()
        num += 1
    
    return losses / num



def val(dataloader, milnet, criterion, device, model='lbmil'):
    milnet.eval()
    losses, num = 0, 0
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for feats, poses, labels in tqdm(dataloader):
            feats, poses, labels = feats.squeeze().to(device), poses.squeeze().to(device), labels.long().to(device)
            if model == 'abmil':
                bag_prediction, _, _ = milnet(feats)
                loss = criterion(bag_prediction, labels)
                y_pred.extend([torch.squeeze(bag_prediction).argmax().cpu().numpy()])
                y_score.extend([torch.squeeze(bag_prediction).softmax(dim=0)[1].cpu().numpy()])
            elif model in ['clam_sb', 'clam_mb']:
                bag_prediction, Y_prob, Y_hat, _, _ = milnet(feats)
                loss = criterion(bag_prediction, labels)
                y_pred.extend([torch.squeeze(Y_hat).cpu().numpy()])
                y_score.extend([torch.squeeze(Y_prob)[1].cpu().numpy()])
            elif model == 'dsmil':
                ins_prediction, bag_prediction, attention, atten_B = milnet(feats)
                max_prediction, _ = torch.max(ins_prediction, 0)
                bag_loss = criterion(bag_prediction, labels)
                max_loss = criterion(max_prediction.unsqueeze(0), labels)
                loss = 0.5*bag_loss + 0.5*max_loss
                y_pred.extend([torch.squeeze(0.5*bag_prediction+0.5*max_prediction).argmax().cpu().numpy()])
                y_score.extend([torch.squeeze(0.5*bag_prediction+0.5*max_prediction).softmax(dim=0)[1].cpu().numpy()])
            elif model == 'transmil':
                output = milnet(feats)
                bag_prediction, Y_prob, Y_hat = output['logits'], output['Y_prob'], output['Y_hat']
                loss = criterion(bag_prediction, labels)
                y_pred.extend([torch.squeeze(Y_hat).cpu().numpy()])
                y_score.extend([torch.squeeze(Y_prob)[1].cpu().numpy()])
            elif model == 'rrtmil':
                bag_prediction = milnet(feats)
                loss = criterion(bag_prediction, labels)
                y_pred.extend([torch.squeeze(bag_prediction).argmax().cpu().numpy()])
                y_score.extend([torch.squeeze(bag_prediction).softmax(dim=0)[1].cpu().numpy()])
            elif model == 'longmil':
                x = torch.cat([feats, poses], dim=1)
                bag_prediction, Y_hat, Y_prob, attention = milnet(x)
                loss = criterion(bag_prediction, labels)
                y_pred.extend([torch.squeeze(Y_hat).cpu().numpy()])
                y_score.extend([torch.squeeze(Y_prob)[1].cpu().numpy()])
            elif model == 'lbmil':
                x = torch.cat([feats, poses], dim=1)
                bag_prediction, Y_hat, Y_prob, attention = milnet(x)
                loss = criterion(bag_prediction, labels)
                y_pred.extend([torch.squeeze(Y_hat).cpu().numpy()])
                y_score.extend([torch.squeeze(Y_prob)[1].cpu().numpy()])

            losses += loss.item()
            num += 1
            y_true.extend(labels.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)

    return losses / num, acc, precision, recall, f1, auc



def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device) if torch.cuda.is_available() else 'cpu'

    train_path, val_path, labels = load_trainval_data(args.data_path, args.label_path)

    trainset = PathologyDataset(train_path, labels)
    valset = PathologyDataset(val_path, labels)

    trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valloader = DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

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
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0.000005)

    extraction = args.data_path.split('/')[-1].split('_')[0] if isinstance(args.data_path, str) else args.data_path[0].split('/')[-1].split('_')[0]
    current_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    save_dir = os.path.join('work_dirs', f'{extraction}_{args.model}', f'{current_time}')
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{current_time}.log', 'w') as f:
        f.write(f'data_path = {args.data_path}\n')
        f.write('extraction = {}, seed = {}, lr = {:.2g}, weight_decay = {:.2g}, epochs = {}\n'.\
            format(extraction, args.seed, args.lr, args.weight_decay, args.epochs))

    min_loss = inf
    for i in range(args.epochs):
        loss = train(trainloader, milnet, criterion, optimizer, device, args.model)
        print('train {}: loss = {:.4f}\n'.format(i+1, loss))
        with open(f'{save_dir}/{current_time}.log', 'a') as file:
            file.write('train {}: loss = {:.4f}\n'.format(i+1, loss))

        val_loss, acc, precision, recall, f1, auc = val(valloader, milnet, criterion, device, args.model)

        print('val {}: loss = {:.4f} | acc = {:.4f} | precision = {:.4f} | recall = {:.4f} | f1 = {:.4f} | auc = {:.4f}\n'.format(i+1, val_loss, acc, precision, recall, f1, auc))
        with open(f'{save_dir}/{current_time}.log', 'a') as file:
            file.write('val {}: loss = {:.4f} | acc = {:.4f} | precision = {:.4f} | recall = {:.4f} | f1 = {:.4f} | auc = {:.4f}\n'.format(i+1, val_loss, acc, precision, recall, f1, auc))

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(milnet.state_dict(), f"{save_dir}/{extraction}_{args.model}.pth")
            print(f'The best checkpoint at epoch {i+1} is saved\n')
            with open(f'{save_dir}/{current_time}.log', 'a') as file:
                file.write(f'The best checkpoint at epoch {i+1} is saved\n')

        scheduler.step()



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--feat_size', default=1536, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    
    parser.add_argument('--model', default='lbmil', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_path', nargs='+', type=str, default=['WSI/features_out/gigapath_features', 'WSI/features_in/gigapath_features'])
    parser.add_argument('--label_path', default='labels/all_labels.xlsx', type=str)
    parser.add_argument('--device', default='cuda:1', type=str)

    args = parser.parse_args()
    main(args)