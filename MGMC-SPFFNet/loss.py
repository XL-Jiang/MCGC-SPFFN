import numpy as np
import torch
def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    HSIC = HSIC/((dim-1)*(dim-1))
    return HSIC

def common_loss(emb1, emb2,emb3):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb3 = emb3 - torch.mean(emb3, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    emb3 = torch.nn.functional.normalize(emb3, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cov3 = torch.matmul(emb3, emb3.t())
    cost1 = torch.mean((cov1 - cov2)**2)
    cost2 = torch.mean((cov3 - cov2) ** 2)
    cost3 = torch.mean((cov1 - cov3) ** 2)
    cost=(cost1+cost2+cost3)/3
    return cost
