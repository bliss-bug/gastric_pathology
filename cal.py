import numpy as np

acc, recall, f1, auc = [], [], [], []

model = 'lbmil'
start = 45

with open('outcome/{}.log'.format(model), 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line[:5] == 'test:':
            acc.append(float(line[28:34]))
            recall.append(float(line[67:73]))
            f1.append(float(line[81:87]))
            auc.append(float(line[96:102]))
    
acc1, acc2, acc3 = acc[start:start+5], acc[start+5:start+10], acc[start+10:start+15]
recall1, recall2, recall3 = recall[start:start+5], recall[start+5:start+10], recall[start+10:start+15]
f11, f12, f13 = f1[start:start+5], f1[start+5:start+10], f1[start+10:start+15]
auc1, auc2, auc3 = auc[start:start+5], auc[start+5:start+10], auc[start+10:start+15]

mean_acc1, std_acc1 = np.mean(acc1), np.std(acc1)
mean_acc2, std_acc2 = np.mean(acc2), np.std(acc2)
mean_acc3, std_acc3 = np.mean(acc3), np.std(acc3)

mean_recall1, std_recall1 = np.mean(recall1), np.std(recall1)
mean_recall2, std_recall2 = np.mean(recall2), np.std(recall2)
mean_recall3, std_recall3 = np.mean(recall3), np.std(recall3)

mean_f11, std_f11 = np.mean(f11), np.std(f11)
mean_f12, std_f12 = np.mean(f12), np.std(f12)
mean_f13, std_f13 = np.mean(f13), np.std(f13)

mean_auc1, std_auc1 = np.mean(auc1), np.std(auc1)
mean_auc2, std_auc2 = np.mean(auc2), np.std(auc2)
mean_auc3, std_auc3 = np.mean(auc3), np.std(auc3)

print('prov-gigapath+{} mean acc = {:.2f}, mean recall = {:.2f}, mean f1 = {:.2f}, mean auc = {:.2f}'.format(model, mean_acc1*100, mean_recall1*100, mean_f11*100, mean_auc1*100))
print('prov-gigapath+{} std acc = {:.2f}, std recall = {:.2f}, std f1 = {:.2f}, std auc = {:.2f}'.format(model, std_acc1*100, std_recall1*100, std_f11*100, std_auc1*100))
print('uni+{} mean acc = {:.2f}, mean recall = {:.2f}, mean f1 = {:.2f}, mean auc = {:.2f}'.format(model, mean_acc2*100, mean_recall2*100, mean_f12*100, mean_auc2*100))
print('uni+{} std acc = {:.2f}, std recall = {:.2f}, std f1 = {:.2f}, std auc = {:.2f}'.format(model, std_acc2*100, std_recall2*100, std_f12*100, std_auc2*100))
print('resnet50+{} mean acc = {:.2f}, mean recall = {:.2f}, mean f1 = {:.2f}, mean auc = {:.2f}'.format(model, mean_acc3*100, mean_recall3*100, mean_f13*100, mean_auc3*100))
print('resnet50+{} std acc = {:.2f}, std recall = {:.2f}, std f1 = {:.2f}, std auc = {:.2f}'.format(model, std_acc3*100, std_recall3*100, std_f13*100, std_auc3*100))
