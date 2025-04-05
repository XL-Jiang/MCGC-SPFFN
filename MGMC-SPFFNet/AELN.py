" Adaptive Edge Learning network "
import torch
from torch.nn import Linear as Lin
from torch import nn
class AELN(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(AELN, self).__init__()
        h1 = 128
        self.parser = nn.Sequential(
            nn.Linear(input_dim, h1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(h1),
            nn.Dropout(dropout),
            nn.Linear(h1, h1, bias=True),

        )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.input_dim = input_dim
        self.model_init()
        self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ReLU()

        self.fc = nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            torch.nn.Linear(128, 128)

        )
        self.gap = nn.AdaptiveAvgPool1d(128)
        self.softmax = nn.Softmax(dim = 1)

    def WG(self, input):
        Mf = self.fc(input)
        Gf = self.gap(input)
        WG = self.softmax(torch.matmul(Gf, Mf.t()))
        return WG

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, X):
        X1 = X[:, 0:self.input_dim]
        X2 = X[:, self.input_dim:]
        WG1 = self.WG(X1)
        WG2 = self.WG(X2)
        WGX1 =  torch.matmul(WG1, X1)
        WGX2 = torch.matmul(WG2, X2)
        h1 = self.parser(WGX1)
        h2 = self.parser(WGX2)
        p = (self.cos(h1, h2) + 1) * 0.5
        return p
