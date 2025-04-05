import numpy as np
import torchmetrics
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import precision_recall_fscore_support,roc_curve, auc
def torchmetrics_accuracy(preds, labels):
    acc = torchmetrics.functional.accuracy(preds, labels,task = 'binary')
    return acc
def torchmetrics_auc(preds, labels):
    auc = torchmetrics.functional.auroc(preds, labels, task="multiclass", num_classes=2)
    return auc
def correct_num(preds, labels):
    """Accuracy, auc with masking.Acc of the masked samples"""
    correct_prediction = np.equal(preds, labels).astype(np.float32)
    return np.sum(correct_prediction)

def prf(preds, labels, is_logit=True):
    p,r,f,s  = precision_recall_fscore_support(labels, preds, average='binary',zero_division=0)
    return [p,r,f]

def plot_embedding(data, label, title):
    plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    p = []
    p2 = []
    for i in range(len(label)):
        if label[i] == 0:
            p.append(plt.scatter(data[i, 0], data[i, 1], lw=0.1, c='#0071C5'))
        elif label[i] == 1:
            p2.append(plt.scatter(data[i, 0], data[i, 1], lw=0.1, c='#DB4437'))
    plt.legend((p[0], p2[0]), ('ASD', 'HC'))
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./figures/{:s}.tiff'.format(title),dpi=100)

def plot_ROC(labels_list, logits_list,auc_list):
    plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'bold'
    color_map = cm.get_cmap('tab10')
    for i in range(len(labels_list)):
        fpr, tpr, _ = roc_curve(labels_list[i], logits_list[i])
        roc_auc = auc(fpr, tpr)
        color = color_map(i)
        plt.plot(fpr, tpr,color = color, label='ROC(AUC = %0.2f)fold %d' % (roc_auc, i+1))
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='AUC=0.5')
    plt.xlabel('False Positive Rate', fontsize=12,fontweight='bold')#,fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12,fontweight='bold')#,fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc="lower right")
    plt.savefig('./figures1/ROC.tiff', dpi=600)
    plt.savefig('./figures1/ROC.eps', format='eps', dpi=10000)
    plt.show()


