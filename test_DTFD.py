import argparse, random
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from data import PathologyDataset
from model.Attention import Attention_Gated as Attention
from model.Attention import Attention_with_Classifier
from model.network import Classifier_1fc, DimReduction
from utils import load_test_data, get_cam_1d


def test(dataloader, classifier, dimReduction, attention, UClassifier, 
        criterion, device, numGroup=3, total_instance=3, distill='AFS'):
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    losses, num = 0, 0
    instance_per_group = total_instance // numGroup

    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for feats, _, labels in tqdm(dataloader):
            feats, labels = feats.squeeze().to(device), labels.long().to(device) # [N, C], [1]

            slide_pseudo_feat = []
            slide_sub_preds = []
            slide_sub_labels = []
            
            feat_index = list(range(feats.shape[0]))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]

            for tindex in index_chunk_list:
                slide_sub_labels.append(labels)
                subFeat_tensor = torch.index_select(feats, dim=0, index=torch.LongTensor(tindex).to(device))
                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
                tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                slide_sub_preds.append(tPredict)

                patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.sigmoid(patch_pred_logits)  ## n x cls

                _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)

                if distill == 'MaxMinS':
                    topk_idx_max = sort_idx[:instance_per_group].long()
                    topk_idx_min = sort_idx[-instance_per_group:].long()
                    topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                    MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                    slide_pseudo_feat.append(MaxMin_inst_feat)
                elif distill == 'MaxS':
                    topk_idx_max = sort_idx[:instance_per_group].long()
                    max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                    slide_pseudo_feat.append(max_inst_feat)
                elif distill == 'AFS':
                    slide_pseudo_feat.append(tattFeat_tensor)

            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

            ## first tier
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
            slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
            loss0 = criterion(slide_sub_preds, slide_sub_labels)

            ## second tier
            gSlidePred = UClassifier(slide_pseudo_feat)
            loss1 = criterion(gSlidePred, labels)

            losses += loss0.item() + loss1.item()
            num += 1

            y_pred.extend([torch.squeeze(gSlidePred).argmax().cpu().numpy()])
            y_score.extend([torch.squeeze(gSlidePred).softmax(dim=0)[1].cpu().numpy()])
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

    classifier = Classifier_1fc(args.feat_size, args.num_classes).to(device)
    attention = Attention(args.feat_size).to(device)
    dimReduction = DimReduction(args.feat_size, args.feat_size, numLayer_Res=0).to(device)
    attCls = Attention_with_Classifier(L=args.feat_size, num_cls=args.num_classes).to(device)

    extraction = args.data_path.split('/')[-1].split('_')[0]
    dic = torch.load("checkpoints/{}_{}.pth".format(extraction, args.model))
    classifier.load_state_dict(dic['classifier'])
    dimReduction.load_state_dict(dic['dim_reduction'])
    attention.load_state_dict(dic['attention'])
    attCls.load_state_dict(dic['att_classifier'])

    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = test(testloader, classifier, dimReduction, attention, attCls, 
                                                                              criterion, device, args.numGroup, args.total_instance, args.distill)
    
    print('extraction = {}, model = {}'.format(extraction, args.model))
    print('test: loss = {:.4f} | acc = {:.4f} | precision = {:.4f} | recall = {:.4f} | f1 = {:.4f} | auc = {:.4f}\n'.format(test_loss, test_acc, test_precision, test_recall, test_f1, test_auc))



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--feat_size', default=1024, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    
    parser.add_argument('--model', default='DTFD', type=str)
    parser.add_argument('--data_path', default='WSI/features/uni_features', type=str)
    parser.add_argument('--label_path', default='labels/NDPI_labels.xlsx', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--distill', default='AFS', type=str)

    parser.add_argument('--numGroup', default=5, type=int)
    parser.add_argument('--total_instance', default=5, type=int)

    args = parser.parse_args()
    main(args)