from typing import List, Dict
import torch.nn as nn 
import torch 
from torch.utils.data import Dataset


# 定義模型
class StockPredictor(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(StockPredictor, self).__init__()
        
        # 确保线性层的 in_features 与输入特征维度匹配
        self.linear_src = nn.Linear(input_dim, embed_dim)  # input_dim 应该是 10
        self.linear_target = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.Transformer(embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, src, target):
        src = self.linear_src(src)
        target = self.linear_target(target)
        transformer_output = self.transformer(src, target)
        output = transformer_output[:, -1, :]
        output = self.fc(output)
        return output


# DataSet
class ModelDataset(Dataset):

    # data loading
    def __init__(self, train,trainMask, label):
        self.train = train
        self.trainMask = trainMask
        self.label = label

    # working for indexing
    def __getitem__(self, index):
        
        return self.train[index], self.trainMask[index], self.label[index]

    # return the length of our dataset
    def __len__(self):
        
        return len(self.train)
