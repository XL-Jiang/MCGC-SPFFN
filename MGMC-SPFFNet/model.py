import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin
import torch_geometric as tg
from AELN import AELN
from metrics import torchmetrics_accuracy as accuracy

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim,output_dim, dropout_prob=0.3):
        super(MultiHeadAttention, self).__init__()

        self.relu = torch.nn.ReLU(inplace=True)
        self.q1 = nn.Linear(input_dim, output_dim)
        self.k1 = nn.Linear(input_dim, output_dim)
        self.v1 = nn.Linear(input_dim, output_dim)

        self.q2 = nn.Linear(input_dim, output_dim)
        self.k2 = nn.Linear(input_dim, output_dim)
        self.v2 = nn.Linear(input_dim, output_dim)

        self.q3 = nn.Linear(input_dim, output_dim)
        self.k3 = nn.Linear(input_dim, output_dim)
        self.v3 = nn.Linear(input_dim, output_dim)

        self.layer_norm_x1 = nn.BatchNorm1d(input_dim)
        self.layer_norm_x2 = nn.BatchNorm1d(input_dim)
        self.layer_norm_x3 = nn.BatchNorm1d(input_dim)

        self.drop = nn.Dropout(dropout_prob)
        self.model_init()
    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
    def forward(self, x1,x2,x3):
        N1, C1 = x1.shape
        N2, C2 = x2.shape
        N3, C3 = x3.shape
        x1 = self.layer_norm_x1(x1)
        x2 = self.layer_norm_x2(x2)
        x3 = self.layer_norm_x3(x3)

        q1 = self.q1(x1).reshape(N1, 1, -1).permute(1, 0, 2)
        q1 = self.drop(q1)
        k1 = self.k1(x1).reshape(N1, 1,-1).permute(1, 0, 2)
        k1 = self.drop(k1)
        v1 = self.v1(x1).reshape(N1, 1, -1).permute(1, 0, 2)
        v1 = self.drop(v1)

        q2 = self.q2(x2).reshape(N2, 1, -1).permute(1, 0, 2)
        q2 = self.drop(q2)
        k2 = self.k2(x2).reshape(N2, 1, -1).permute(1, 0, 2)
        k2 = self.drop(k2)
        v2 = self.v2(x2).reshape(N2, 1, -1).permute(1, 0, 2)
        v2 = self.drop(v2)

        q3 = self.q3(x3).reshape(N3, 1, -1).permute(1, 0, 2)
        q3 = self.drop(q3)
        k3 = self.k3(x3).reshape(N3, 1,-1).permute(1, 0, 2)
        k3 = self.drop(k3)
        v3 = self.v3(x3).reshape(N3, 1, -1).permute(1, 0, 2)
        v3 = self.drop(v3)

        qk1 = q1 * k1
        qk1 = torch.softmax(qk1, dim=-1)
        qk2 = q2 * k2
        qk2 = torch.softmax(qk2, dim=-1)
        qk3 = q3 * k3
        qk3 = torch.softmax(qk3, dim=-1)
        att12 = qk1 * v2
        att13 = qk1 * v3
        att21 = qk2 * v1
        att23 = qk2 * v3
        att31 = qk3 * v1
        att32 = qk3 * v2
        att1 = att12 + att13
        att2 = att21 + att23
        att3 = att31 + att32
        att1 = att1.reshape(N1, -1)
        att2 = att2.reshape(N2, -1)
        att3 = att3.reshape(N3, -1)
        att1 = torch.add(att1, x1)
        att2 = torch.add(att2, x2)
        att3 = torch.add(att3, x3)
        qk = torch.cat([att1, att2, att3], dim=-1)
        return qk
class AELNGCN(nn.Module):
    def __init__(self, input_dim, nhid,ngl, dropout,  edgenet_input_dim):
        super(AELNGCN, self).__init__()
        self.ChebConv = tg.nn.ChebConv
        hidden = [nhid for i in range(ngl)]
        self.dropout = dropout
        self.relu = torch.nn.ReLU(inplace=True)
        self.ngl = ngl
        self.nhid = nhid
        self.gconv = nn.ModuleList()
        for i in range(ngl):
            in_channels = input_dim if i == 0 else hidden[i - 1]
            self.gconv.append(self.ChebConv(in_channels, hidden[i], K=3, normalization='sym', bias=True))
        self.model_init()
        self.edge_net = AELN(input_dim=edgenet_input_dim // 2, dropout=dropout)
    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, features, edge_index, edgenet_input):
        edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        # GCN with residual connection
        features = F.dropout(features, self.dropout, self.training)
        x = self.relu(self.gconv[0](features, edge_index, edge_weight))
        x_temp = x
        for i in range(1, self.ngl - 1): # self.ngl→7
            x = F.dropout(x_temp, self.dropout, self.training)
            x = self.relu(self.gconv[i](x, edge_index, edge_weight))
            x_temp = x + x_temp
        x = F.dropout(x_temp, self.dropout, self.training)
        x = self.relu(self.gconv[self.ngl - 1](x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, self.training)
        output = x
        return output
class MCGCSPFFNet(nn.Module):
    def __init__(self, input_dim,nhid,ngl,nclass,dropout,edgenet_input_dim):
        super(MCGCSPFFNet, self).__init__()
        self.AELNGCN1 = AELNGCN(input_dim,nhid,ngl,dropout,edgenet_input_dim)
        self.AELNGCN2 = AELNGCN(input_dim,nhid,ngl,dropout,edgenet_input_dim)
        self.AELNGCN3 = AELNGCN(input_dim,nhid,ngl,dropout,edgenet_input_dim)
        self.MultiHeadAttention = MultiHeadAttention(nhid,24,dropout)
        self.MLP = nn.Sequential(
            torch.nn.Linear(nhid, nhid),
            nn.BatchNorm1d(nhid),
            torch.nn.Linear(nhid, nclass),
            nn.Softmax(dim=1)
        )
        self.cls = nn.Sequential(
            nn.Linear(nhid*3, nhid),
            torch.nn.Linear(nhid, 2),
            nn.Softmax(dim=1)
        )
    def get_acc_weight(self,ind,labels,predications1,predications2,predications3,predications4,predications5,predications6):
        predications1 = self.MLP(predications1)
        predications2 = self.MLP(predications2)
        predications3 = self.MLP(predications3)
        predications4 = self.MLP(predications4)
        predications5 = self.MLP(predications5)
        predications6 = self.MLP(predications6)
        logits_test1 = predications1[ind].detach().cpu().numpy()
        correct1, acc1 = accuracy(logits_test1, labels[ind])
        logits_test2 = predications2[ind].detach().cpu().numpy()
        correc2, acc2 = accuracy(logits_test2, labels[ind])
        logits_test3 = predications3[ind].detach().cpu().numpy()
        correct3, acc3 = accuracy(logits_test3, labels[ind])
        logits_test4 = predications4[ind].detach().cpu().numpy()
        correct4, acc4 = accuracy(logits_test4, labels[ind])
        logits_test5 = predications5[ind].detach().cpu().numpy()
        correct5, acc5 = accuracy(logits_test5, labels[ind])
        logits_test6 = predications6[ind].detach().cpu().numpy()
        correct6, acc6 = accuracy(logits_test6, labels[ind])
        return acc1,acc2,acc3,acc4,acc5,acc6

    def forward(self, ind,labels,intput_ftr1,intput_ftr2,intput_ftr3, edge_index1,edgenet_input1,edge_index2,edgenet_input2,edge_index3,edgenet_input3):
        output1 = self.AELNGCN1(intput_ftr1,edge_index3,edgenet_input3)
        output2 = self.AELNGCN1(intput_ftr1,edge_index2,edgenet_input2)

        output3 = self.AELNGCN2(intput_ftr2, edge_index1, edgenet_input1)
        output4 = self.AELNGCN2(intput_ftr2, edge_index3, edgenet_input3)

        output5 = self.AELNGCN3(intput_ftr3, edge_index2, edgenet_input2)
        output6 = self.AELNGCN3(intput_ftr3, edge_index1, edgenet_input1)

        labels = labels.detach().cpu().numpy()
        w1,w2,w3,w4,w5,w6 = self.get_acc_weight(ind,labels,output1,output2,output3,output4,output5,output6)
        sum_w = w1+w2+w3+w4+w5+w6
        # Accuracy-weighted voting strategy
        coss1 = ((w1+w2)/sum_w)*(output1+output2)
        coss2 = ((w3+w4)/sum_w)*(output3+output4)
        coss3 = ((w5+w6)/sum_w)*(output5+output6)
        #Multi-head cross-attention mechanism
        cross_output = self.MultiHeadAttention(coss1,coss2,coss3)
        output_com = cross_output
        #Final features
        output = self.cls(output_com)
        #Multi-scale features
        coss1 = self.MLP(coss1)
        coss2 = self.MLP(coss2)
        coss3 = self.MLP(coss3)
        # Multi-channel features
        output1 = self.MLP(output1)
        output2= self.MLP(output2)
        output3 = self.MLP(output3)
        output4 = self.MLP(output4)
        output5 = self.MLP(output5)
        output6 = self.MLP(output6)
        return output,output1,output2,output3,output4,output5,output6,coss1,coss2,coss3