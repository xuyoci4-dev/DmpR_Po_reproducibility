import torch
from torch import nn
class MLP880(nn.Module):
    def __init__(self,dropout=.25):
        super().__init__(); widths=[880,512,256,128,64]; layers=[]
        for source,target in zip(widths[:-1],widths[1:]): layers += [nn.Linear(source,target),nn.BatchNorm1d(target),nn.GELU(),nn.Dropout(dropout)]
        self.network=nn.Sequential(*layers,nn.Linear(64,1))
    def forward(self,x): return self.network(x).squeeze(-1)
class ResidualBlock(nn.Module):
    def __init__(self,source,target,kernel,dropout):
        super().__init__(); padding=kernel//2
        self.body=nn.Sequential(nn.Conv1d(source,target,kernel,padding=padding),nn.BatchNorm1d(target),nn.ReLU(),nn.Dropout(dropout),nn.Conv1d(target,target,kernel,padding=padding),nn.BatchNorm1d(target))
        self.skip=nn.Identity() if source==target else nn.Conv1d(source,target,1); self.activation=nn.ReLU()
    def forward(self,x): return self.activation(self.body(x)+self.skip(x))
class ResidualCNN(nn.Module):
    def __init__(self,dropout=.25):
        super().__init__(); self.stem=nn.Sequential(nn.Conv1d(4,32,7,padding=3),nn.BatchNorm1d(32),nn.ReLU())
        self.network=nn.Sequential(ResidualBlock(32,64,5,dropout),nn.MaxPool1d(2),ResidualBlock(64,128,5,dropout),nn.MaxPool1d(2),ResidualBlock(128,128,3,dropout),nn.AdaptiveAvgPool1d(1))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(128,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,1))
    def forward(self,x): return self.head(self.network(self.stem(x))).squeeze(-1)
