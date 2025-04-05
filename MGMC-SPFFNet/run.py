import time
import warnings
from model import MCGCSPFFNet
from loss import loss_dependence, common_loss
from dataloader import *
from opt import *
from metrics import torchmetrics_accuracy, torchmetrics_auc, prf, correct_num,plot_ROC
from sklearn.metrics import roc_curve,auc
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    opt = OptInit().initialize()
    opt.cuda = not opt.no_cuda and torch.cuda.is_available()
    print('  Loading ABIDE dataset ...')
    dl = dataloader()
    raw_features1,raw_features2,raw_features3, y, nonimg = dl.load_data()
    cv_splits = dl.data_3split(opt.folds)
    corrects = np.zeros(opt.folds, dtype=np.int32)
    accs = np.zeros(opt.folds, dtype=np.float32)
    aucs = np.zeros(opt.folds, dtype=np.float32)
    prfs = np.zeros([opt.folds, 3], dtype=np.float32)  ## Save Precision, Recall, F1
    test_num = np.zeros(opt.folds, dtype=np.float32)
    t_start = time.time()

    if opt.train == 0:
        y_label_list = []
        y_pre_list = []
        print("\r\n=====Strat Train & Val =====")
        for fold in range(opt.folds):
            print("\r\n========================== Fold {} ==========================".format(fold+1))
            train_ind = cv_splits[fold][0]
            test_ind = cv_splits[fold][1]
            if torch.cuda.is_available():
                torch.cuda.manual_seed(opt.seed)  # cuda
            np.random.seed(opt.seed)
            random.seed(opt.seed)

            #get features
            node_ftr1, node_ftr2, node_ftr3, edge_ftr1, edge_ftr2, edge_ftr3 = dl.get_node_edge_features(opt.ftr_dim,train_ind)
            # get AELN inputs
            edge_index1, edgenet_input1 = dl.get_AELN_inputs(edge_ftr1)
            edge_index2, edgenet_input2 = dl.get_AELN_inputs(edge_ftr2)
            edge_index3, edgenet_input3 = dl.get_AELN_inputs(edge_ftr3)
            # normalization
            edgenet_input1 = (edgenet_input1 - edgenet_input1.mean(axis=0)) / edgenet_input1.std(axis=0)
            edgenet_input2 = (edgenet_input2 - edgenet_input2.mean(axis=0)) / edgenet_input2.std(axis=0)
            edgenet_input3 = (edgenet_input3 - edgenet_input3.mean(axis=0)) / edgenet_input3.std(axis=0)

            model = MCGCSPFFNet(
                input_dim=opt.ftr_dim,
                nhid = opt.units,
                ngl = opt.lg,
                nclass=opt.nclass,
                dropout=opt.dropout,
                edgenet_input_dim=2 * opt.ftr_dim
            )
            model = model.to(opt.device)
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)

            if opt.cuda:
                intput_ftr1 = torch.tensor(node_ftr1, dtype=torch.float32).cuda()
                intput_ftr2 = torch.tensor(node_ftr2, dtype=torch.float32).cuda()
                intput_ftr3 = torch.tensor(node_ftr3, dtype=torch.float32).cuda()
                edge_index1 = torch.tensor(edge_index1, dtype=torch.long).cuda()
                edgenet_input1 = torch.tensor(edgenet_input1, dtype=torch.float32).cuda()
                edge_index2 = torch.tensor(edge_index2, dtype=torch.long).cuda()
                edgenet_input2 = torch.tensor(edgenet_input2, dtype=torch.float32).cuda()
                edge_index3 = torch.tensor(edge_index3, dtype=torch.long).cuda()
                edgenet_input3 = torch.tensor(edgenet_input3, dtype=torch.float32).cuda()
                labels = torch.tensor(y, dtype=torch.long).cuda()
                fold_model_path = './save' + "/fold{}.pth".format(fold + 1)

            print("  Start Training")
            print('  Train samples: ', len(train_ind) , 'Val samples: ', len(test_ind))

            ## Save Precision, Recall, F1
            prfs = np.zeros([opt.folds, 3], dtype=np.float32)
            train_loss_all = []
            val_loss_all = []
            best_val_loss = float('inf')
            current_patience = 0

            for epoch in range(opt.epoch):
                model.train()
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    output,output1,output2,output3,output4,output5,output6,coss1,coss2,coss3 = model(train_ind,labels,intput_ftr1, intput_ftr2,intput_ftr3,edge_index1, edgenet_input1, edge_index2, edgenet_input2,edge_index3, edgenet_input3)

                    loss_class = loss_fn(output[train_ind], labels[train_ind])
                    #channel diversity constraints
                    loss_dep = (loss_dependence(output1, output2, raw_features1.shape[0])
                                + loss_dependence(output3, output4, raw_features2.shape[0])
                                + loss_dependence(output5, output6, raw_features3.shape[0]))/ 3
                    #scale correlation  constraints
                    loss_com = common_loss(coss1,coss2,coss3)
                    loss = loss_class + 5e-5* loss_dep + 1e-3 * loss_com
                    loss_class.backward()
                    optimizer.step()

                loss_train = loss_class
                train_loss_all.append(loss_train.data.cpu().numpy())
                pred_labels = output.argmax(dim=1)
                acc_train = torchmetrics_accuracy(pred_labels[train_ind], labels[train_ind])
                auc_train = torchmetrics_auc(output[train_ind], labels[train_ind])
                logits_train = pred_labels[train_ind].detach().cpu().numpy()
                prf_train = prf(logits_train, y[train_ind])
                print('Train_epoch:', epoch,
                      'Train_acc:{:.4f}'.format(acc_train),
                      'Train_pre:{:.4f}'.format(prf_train[0]),
                      'Train_recall:{:.4f}'.format(prf_train[1]),
                      'Train_F1:{:.4f}'.format(prf_train[2]),
                      'Train_AUC:{:.4f}'.format(auc_train))

                model.eval()
                with torch.set_grad_enabled(False):
                    output, output1, output2, output3, output4, output5, output6,coss1,coss2,coss3 = model(train_ind, labels, intput_ftr1, intput_ftr2, intput_ftr3,edge_index1, edgenet_input1,edge_index2, edgenet_input2,edge_index3, edgenet_input3)

                loss_val = loss_fn(output[test_ind], labels[test_ind])
                val_loss_all.append(loss_val.data.cpu().numpy())
                pred_labels = output.argmax(dim=1)
                auc_val = torchmetrics_auc(output[test_ind], labels[test_ind])
                logits_val = pred_labels.detach().cpu().numpy()
                prf_val = prf(logits_val[test_ind],y[test_ind])
                correct_val = correct_num(logits_val[test_ind], y[test_ind])
                acc_val = correct_val / len(test_ind)
                label = labels.detach().cpu().numpy()
                y_label_list.append(label.tolist())
                y_pre_list.append(logits_val.tolist())
                print('Valid_epoch:',epoch,
                    'val_acc:{:.4f}'.format(acc_val),
                      'val_pre:{:.4f}'.format(prf_val[0]),
                      'val_recall:{:.4f}'.format(prf_val[1]),
                      'val_F1:{:.4f}'.format(prf_val[2]),
                      'val_AUC:{:.4f}'.format(auc_val))

                #early stopping
                if epoch > 120 and opt.early_stopping == True:
                    if loss_val < best_val_loss:
                        best_val_loss = loss_val
                        current_patience = 0
                    else:
                        current_patience += 1
                    if current_patience >= opt.early_stopping_patience:
                        torch.save(model.state_dict(), fold_model_path)
                        print('  Early Stopping!!! epoch：{}'.format(epoch))
                        break

        print("=================>Train Done<======================")

    if opt.train==1:
            lable_list = []
            pre_list = []
            auc_list = []
            fprs = np.zeros([opt.folds], dtype=np.float32)
            tprs = np.zeros([opt.folds], dtype=np.float32)

            for fold in range(opt.folds):
                print("\r\n========================== Fold {} ==========================".format(fold + 1))
                i = fold
                train_ind = cv_splits[i][0]
                test_ind = cv_splits[i][1]
                print("\r\n==>Loading the Model for the {}-th Fold:... ...".format(fold + 1),
                      "Size of samples in the test set:{}".format(len(test_ind)))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(opt.seed)  # cuda
                np.random.seed(opt.seed)
                random.seed(opt.seed)
                # get features
                node_ftr1, node_ftr2, node_ftr3, edge_ftr1, edge_ftr2, edge_ftr3 = dl.get_node_edge_features(
                    opt.ftr_dim, train_ind)
                # get AELN inputs
                edge_index1, edgenet_input1 = dl.get_AELN_inputs(edge_ftr1)
                edge_index2, edgenet_input2 = dl.get_AELN_inputs(edge_ftr2)
                edge_index3, edgenet_input3 = dl.get_AELN_inputs(edge_ftr3)
                # normalization
                edgenet_input1 = (edgenet_input1 - edgenet_input1.mean(axis=0)) / edgenet_input1.std(axis=0)
                edgenet_input2 = (edgenet_input2 - edgenet_input2.mean(axis=0)) / edgenet_input2.std(axis=0)
                edgenet_input3 = (edgenet_input3 - edgenet_input3.mean(axis=0)) / edgenet_input3.std(axis=0)

                model = MCGCSPFFNet(
                    input_dim=opt.ftr_dim,
                    nhid=opt.units,
                    ngl=opt.lg,
                    nclass=opt.nclass,
                    dropout=opt.dropout,
                    edgenet_input_dim=2 * opt.ftr_dim
                )
                model = model.to(opt.device)

                if opt.cuda:
                    intput_ftr1 = torch.tensor(node_ftr1, dtype=torch.float32).cuda()
                    intput_ftr2 = torch.tensor(node_ftr2, dtype=torch.float32).cuda()
                    intput_ftr3 = torch.tensor(node_ftr3, dtype=torch.float32).cuda()
                    edge_index1 = torch.tensor(edge_index1, dtype=torch.long).cuda()
                    edgenet_input1 = torch.tensor(edgenet_input1, dtype=torch.float32).cuda()
                    edge_index2 = torch.tensor(edge_index2, dtype=torch.long).cuda()
                    edgenet_input2 = torch.tensor(edgenet_input2, dtype=torch.float32).cuda()
                    edge_index3 = torch.tensor(edge_index3, dtype=torch.long).cuda()
                    edgenet_input3 = torch.tensor(edgenet_input3, dtype=torch.float32).cuda()
                    labels = torch.tensor(y, dtype=torch.long).cuda()
                    fold_model_path = './save' + "/fold{}.pth".format(fold + 1)
                model.load_state_dict(torch.load(fold_model_path))
                model.eval()
                with torch.set_grad_enabled(False):
                    output, output1, output2, output3, output4, output5, output6,coss1,coss2,coss3 = model(train_ind, labels, intput_ftr1,intput_ftr2, intput_ftr3,edge_index1, edgenet_input1,edge_index2, edgenet_input2,edge_index3, edgenet_input3)
                pre_labels = output.argmax(dim=1)
                acc_test = torchmetrics_accuracy(pre_labels[test_ind], labels[test_ind])
                pres = np.max(output.detach().cpu().numpy(), axis=1)
                labels = labels.detach().cpu()
                logits_test = pre_labels.detach().cpu().numpy()
                prf_test = prf(logits_test[test_ind], y[test_ind])
                correct_test = correct_num(logits_test[test_ind], y[test_ind])
                fpr, tpr, _ = roc_curve(labels[test_ind], logits_test[test_ind])
                fprs[fold]=fpr[1]
                tprs[fold]=tpr[1]

                roc_auc = auc(fpr, tpr)
                auc_test = roc_auc
                #Draw ROC
                lable_list.append(labels[test_ind])
                pre_list.append(logits_test[test_ind])
                auc_list.append(auc_test)
                plot_ROC(lable_list,pre_list,auc_list)

                t_end = time.time()
                t = t_end - t_start
                print('==>Test in Fold {} Results:'.format(fold + 1),
                      'test acc:{:.4f}'.format(acc_test),
                      'test_pre:{:.4f}'.format(prf_test[0]),
                      'test_recall:{:.4f}'.format(prf_test[1]),
                      'test_F1:{:.4f}'.format(prf_test[2]),
                      'test_AUC:{:.4f}'.format(auc_test),
                      'time:{:.3f}s'.format(t))
                aucs[fold] = auc_test
                prfs[fold] = prf_test
                corrects[fold] = correct_test
                test_num[fold] = len(test_ind)

            print("\r\n===============Nested10kCV==================")
            Ten_fold_cross_validation_acc= np.sum(corrects) / np.sum(test_num)
            #print the results of every fold
            print("======== 10 Folds ACC ======")
            for i in range(opt.folds):
                fold_acc = corrects[i] / test_num[i]
                print(fold_acc)
            print("======== 10 Folds Precision ======")
            for i in range(opt.folds):
                fold_Pre = prfs[i,0]
                print(fold_Pre)
            print("======== 10 Folds ARecall ======")
            for i in range(opt.folds):
                fold_recall = prfs[i,1]
                print(fold_recall)
            print("======== 10 Folds F1-score ======")
            for i in range(opt.folds):
                fold_f1 = prfs[i,2]
                print(fold_f1)
            print("======== 10 Folds AUC ======")
            for i in range(opt.folds):
                fold_auc = aucs[i]
                print(fold_auc)
            print("======== 10 Folds FPR ======")
            for i in range(opt.folds):
                fold_fpr = fprs[i]
                print(fold_fpr)
            print("======== 10 Folds TPR ======")
            for i in range(opt.folds):
                fold_tpr = tprs[i]
                print(fold_tpr)

            Ten_fold_cross_validation_auc = np.mean(aucs)
            Ten_fold_cross_validation_precision = np.mean(prfs[:,0])
            Ten_fold_cross_validation_recall = np.mean(prfs[:,1])
            Ten_fold_cross_validation_F1 = np.mean(prfs[:,2])
            print("=> Average test accuracy in {}-fold CV: {:.5f}".format(opt.folds, Ten_fold_cross_validation_acc))
            print("=> Average test AUC in  {}-fold CV: {:.5f}".format(opt.folds, Ten_fold_cross_validation_auc))
            print("=> Average test precision {:.5f}, recall {:.5f}, F1-score {:.5f}".format(
                Ten_fold_cross_validation_precision, Ten_fold_cross_validation_recall, Ten_fold_cross_validation_F1))






