import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from data import PathologyDataset
from model import abmil, clam, dsmil, transmil, rrt, longmil, lbmil
from utils import load_test_data


def test(dataloader, milnet, criterion, device, model='abmil'):
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
    cm = confusion_matrix(y_true, y_pred)

    return losses / num, acc, precision, recall, f1, auc



def main(args):
    device = torch.device(args.device) if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()

    test_path, labels = load_test_data(args.data_path, args.label_path)
    testset = PathologyDataset(test_path, labels)
    testloader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.model == 'abmil':
        milnet = abmil.Attention(in_size=args.feat_size, out_size=args.num_classes).to(device)
    elif args.model == 'clam_sb':
        milnet = clam.CLAM_SB(dropout=0.25, n_classes=args.num_classes, embed_dim=args.feat_size).to(device)
    elif args.model == 'clam_mb':
        milnet = clam.CLAM_MB(dropout=0.25, n_classes=args.num_classes, embed_dim=args.feat_size).to(device)
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

    extraction = args.data_path.split('/')[-1].split('_')[0]
    milnet.load_state_dict(torch.load("checkpoints/{}_{}.pth".format(extraction, args.model)))
    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = test(testloader, milnet, criterion, device, args.model)
    
    print('extraction = {}, model = {}'.format(extraction, args.model))
    print('test: loss = {:.4f} | acc = {:.4f} | precision = {:.4f} | recall = {:.4f} | f1 = {:.4f} | auc = {:.4f}\n'.format(test_loss, test_acc, test_precision, test_recall, test_f1, test_auc))



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--feat_size', default=1024, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    
    parser.add_argument('--model', default='abmil', type=str)
    parser.add_argument('--data_path', default='WSI/features/uni_features', type=str)
    parser.add_argument('--label_path', default='labels/NDPI_labels.xlsx', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)

    args = parser.parse_args()
    main(args)