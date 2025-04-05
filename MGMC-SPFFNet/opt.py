import datetime
import argparse
import random
import numpy as np
import torch
class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation')
        parser.add_argument('--train', default=0, type=int, help='train(default) or evaluate')
        parser.add_argument('--ftr_dim', default=1500, type=int, help='The dimension of features')
        parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
        parser.add_argument('--folds', default=10, type=int,help='For cross validation, specifies which fold will be used. All folds are used if set to 11 (default: 11)')
        parser.add_argument('--early_stopping', action='store_true', default=False, help='early stopping switch')
        parser.add_argument('--early_stopping_patience', type=int, default=20, help='early stoppng epochs')
        parser.add_argument('--units', type=int, default=24, help='hidden units of gconv layer')
        parser.add_argument('--lg', type=int, default=4, help='number of gconv layers')
        parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
        parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
        parser.add_argument('--epoch', default=150, type=int, help='number of epochs for training')
        parser.add_argument('--dropout', default=0.3, type=float, help='ratio of dropout')
        parser.add_argument('--nclass', type=int, default=2, help='number of classes')
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        args = parser.parse_args()
        args.time = datetime.datetime.now().strftime("%y%m%d")
        if args.no_cuda:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(" Using GPU in torch")
        self.args = args

    def print_args(self):
        # self.args.printer args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")
        phase = 'train' if self.args.train==1 else 'eval'
        print('===> Phase is {}.'.format(phase))

    def initialize(self):
        self.set_seed(123)
        self.print_args()
        return self.args

    def set_seed(self, seed=123):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


