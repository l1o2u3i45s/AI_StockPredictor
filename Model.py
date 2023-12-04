from typing import List, Dict
import torch.nn as nn 
import torch 
from torch.utils.data import Dataset
import torch.nn.functional as F


#LSTM模型
class LSTM(nn.Module):
    def __init__(self, dimension):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=dimension, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.linear_src = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),  # 使用 ReLU 代替 Sigmoid
            nn.Dropout(0.5),  # 添加 Dropout 層
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        x = out[:, -1, :]        
        x = self.linear_src(x) 
        x = torch.sigmoid(x)  # 對於二分類問題使用 Sigmoid
       
        return x
 
        
    
# 定義模型
class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(Transformer, self).__init__()
        
        # 确保线性层的 in_features 与输入特征维度匹配
        self.linear_src = nn.Sequential(
          nn.Linear(input_dim, embed_dim),
          nn.Sigmoid(),
        ) 
        self.linear_target = nn.Sequential(
          nn.Linear(input_dim, embed_dim),
          nn.Sigmoid(),
        ) 
        self.transformer = nn.Transformer(embed_dim, batch_first=True)

        self.fc = nn.Sequential(
          nn.Linear(embed_dim, 16),
          nn.Sigmoid(),
          nn.Linear(16, 2)
        )  
        self.softmax = nn.Softmax(dim=1)

        

    def forward(self, src, target):
        srcData = self.linear_src(src)
        targetData = self.linear_target(target)
        transformer_output = self.transformer(srcData, targetData)
        output = transformer_output[:, -1, :]
        output = self.fc(output) 
        output = self.softmax(output)
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
