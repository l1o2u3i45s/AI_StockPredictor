from typing import List, Dict
import torch.nn as nn 
import torch 
from torch.utils.data import Dataset


# 定義模型
class StockPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StockPredictor, self).__init__()
        
        d_model = input_dim
        nhead = input_dim // 2
        if d_model % nhead != 0:
            raise ValueError("input_dim must be divisible by nhead")
        
        num_encoder_layers = 6
        dim_feedforward = 2048
        dropout = 0.1
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        transformer_output = self.transformer_encoder(x).squeeze(1)
        final_output = self.fc(transformer_output)
        return final_output


# DataSet
class ModelDataset(Dataset):

    # data loading
    def __init__(self, train, label):
        self.train = train
        self.label = label

    # working for indexing
    def __getitem__(self, index):
        
        return self.train[index], self.label[index]

    # return the length of our dataset
    def __len__(self):
        
        return len(self.train)
