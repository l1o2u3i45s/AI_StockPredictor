from typing import List, Dict
import torch.nn as nn 
import torch 
from torch.utils.data import Dataset

#LSTM模型
class LSTM(nn.Module):
    def __init__(self,dimension):
        super(LSTM,self).__init__()
        self.lstm=nn.LSTM(input_size=dimension,hidden_size=128,num_layers=3,batch_first=True)
        self.linear1=nn.Linear(in_features=128,out_features=16)
        self.linear2=nn.Linear(16,2)
    def forward(self,x):
        out,_=self.lstm(x)
        x=out[:,-1,:]        
        x=self.linear1(x)
        x=self.linear2(x)
        return x
        
    
# 定義模型
class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(Transformer, self).__init__()
        
        # 确保线性层的 in_features 与输入特征维度匹配
        self.linear_src = nn.Linear(input_dim, embed_dim)  # input_dim 应该是 10
        self.linear_target = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.Transformer(embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, src, target):
        srcData = self.linear_src(src)
        targetData = self.linear_target(target)
        transformer_output = self.transformer(srcData, targetData)
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
