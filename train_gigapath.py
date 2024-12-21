import os
import random
import argparse
from math import inf
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from gigapath.classification_head import ClassificationHead
from data import PathologyDataset
from utils import load_data


def train(dataloader, model, criterion, optimizer, device):
    model.train()
    losses, num = 0, 0

    for feats, poses, labels in tqdm(dataloader):
        optimizer.zero_grad()
        feats, poses, labels = feats.to(device), poses.to(device), labels.long().to(device)
        logits = model(feats, poses)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses += loss.item()
        num += 1

    return losses / num



def val(dataloader, model, criterion, device):
    model.eval()
    losses, num = 0, 0
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for feats, poses, labels in tqdm(dataloader):
            feats, poses, labels = feats.to(device), poses.to(device), labels.long().to(device)
            logits = model(feats, poses)
            loss = criterion(logits, labels)

            y_pred.extend([torch.squeeze(logits).argmax().cpu().numpy()])
            y_score.extend([torch.squeeze(logits).softmax(dim=0)[1].cpu().numpy()])
            y_true.extend(labels.cpu().numpy())
            losses += loss.item()
            num += 1

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
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device) if torch.cuda.is_available() else 'cpu'

    model = ClassificationHead(args.input_dim, args.latent_dim, args.feat_layer, args.num_classes,
                               pretrained=args.pretrained).to(device)

    train_path, val_path, test_path, labels = load_data(args.data_path, args.label_path, args.fold)
    trainset = PathologyDataset(train_path, labels)
    valset = PathologyDataset(val_path, labels)
    testset = PathologyDataset(test_path, labels)

    trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valloader = DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    testloader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs('outcome', exist_ok=True)
    os.makedirs('best_checkpoints', exist_ok=True)

    extraction = args.data_path[0].split('/')[-1].split('_')[0]
    with open('outcome/gigapath.log', 'a+') as file:
        file.write('extraction = {}, seed = {}, fold = {}, lr = {:.2g}, weight_decay = {:.2g}, epochs = {}\n'.\
                   format(extraction, args.seed, args.fold, args.lr, args.weight_decay, args.epochs))

    min_loss = inf
    for i in range(args.epochs):
        loss = train(trainloader, model, criterion, optimizer, device)
        print('train {}: loss = {:.4f}\n'.format(i+1, loss))
        with open('outcome/gigapath.log', 'a+') as file:
            file.write('train {}: loss = {:.4f}\n'.format(i+1, loss))

        val_loss, acc, precision, recall, f1, auc = val(valloader, model, criterion, device)
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), f"best_checkpoints/gigapath_slide_fold{args.fold}.pth")

        print('val {}: loss = {:.4f} | acc = {:.4f} | precision = {:.4f} | recall = {:.4f} | f1 = {:.4f} | auc = {:.4f}\n'.format(i+1, val_loss, acc, precision, recall, f1, auc))
        with open('outcome/gigapath.log', 'a+') as file:
            file.write('val {}: loss = {:.4f} | acc = {:.4f} | precision = {:.4f} | recall = {:.4f} | f1 = {:.4f} | auc = {:.4f}\n'.format(i+1, val_loss, acc, precision, recall, f1, auc))

    model = ClassificationHead(args.input_dim, args.latent_dim, args.feat_layer, args.num_classes,
                               pretrained=args.pretrained).to(device)
    model.load_state_dict(torch.load(f"best_checkpoints/gigapath_slide_fold{args.fold}.pth"))

    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = val(testloader, model, criterion, device)
    print('test: loss = {:.4f} | acc = {:.4f} | precision = {:.4f} | recall = {:.4f} | f1 = {:.4f} | auc = {:.4f}\n'.format(test_loss, test_acc, test_precision, test_recall, test_f1, test_auc))
    with open('outcome/gigapath.log', 'a+') as file:
        file.write('test: loss = {:.4f} | acc = {:.4f} | precision = {:.4f} | recall = {:.4f} | f1 = {:.4f} | auc = {:.4f}\n\n'.format(test_loss, test_acc, test_precision, test_recall, test_f1, test_auc))



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--input_dim', default=1536, type=int)
    parser.add_argument('--latent_dim', default=768, type=int)
    parser.add_argument('--feat_layer', default='5-11', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    
    parser.add_argument('--seed', default=460, type=int)
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--data_path', nargs='+', type=str, default=['WSI/features/gigapath_features', 'WSI/features2/gigapath_features', 'WSI/features3/gigapath_features'])
    parser.add_argument('--label_path', default='labels/NDPI_labels.xlsx', type=str)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--pretrained', default='checkpoints/slide_encoder.pth', type=str)

    args = parser.parse_args()
    main(args)
