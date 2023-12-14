import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import Model
import DataService
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
  
tensors, labels = DataService.GetData()
input_DModel = DataService.GetDModel()

trainDatasize = int(len(tensors) * 0.8)
 

trainTensors = tensors[0 : trainDatasize] 
trainLabelTensor = labels[0 : trainDatasize] 
testTensors = tensors[trainDatasize+1 : len(tensors)] 
testLabelTensor = labels[trainDatasize+1 : len(labels)] 
# 轉換為PyTorch張量
window_size = DataService.GetWindowSize()
maskTensor = DataService.GetMaskData()

trainData = [torch.stack(trainTensors[i:i+window_size]) for i in range(len(trainTensors) - window_size)]
trainMaskData = [torch.stack(trainTensors[i:i+window_size]+ [maskTensor]) for i in range(len(trainTensors) - window_size)]
trainLabel = trainLabelTensor[window_size : len(trainLabelTensor)]

trainDataSet = Model.ModelDataset(trainData,trainMaskData, trainLabel) 


# # 實例化模型、損失函數和優化器
trainType = 2

if trainType == 1: #TransFormer
    train_loader = DataLoader(trainDataSet, shuffle=True, batch_size=16)
    model = Model.Transformer(input_dim= input_DModel).to(device) 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # # 訓練模型
    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        print(epoch)
        total_loss = 0
        num_batches = 0
        for inputs, inputMask, labels in train_loader:
            inputs,inputMask, labels = inputs.to(device)    ,inputMask.to(device)     , labels.to(device)  
 
            optimizer.zero_grad()
            outputs = model(inputs,inputMask)
             
            loss = criterion(outputs, labels)
            #print("Predict OpenPrice:", outputs[0, 0].item(), "Predict ClosePrice:", outputs[0, 1].item())
            loss.backward() 
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f"Epoch {epoch} - Average Loss: {average_loss:.4f}")

    # # 預測並獲取預測值
    model.eval()
    torch.save(model.state_dict(), './TransFormer.pth')

elif trainType == 2: #LSTM
    train_loader = DataLoader(trainDataSet, shuffle=True, batch_size=16)
    model = Model.LSTM(dimension = input_DModel).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # # 訓練模型
    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        print(epoch)
        for inputs, inputMask, labels in train_loader:
            inputs,inputMask, labels = inputs.to(device),inputMask.to(device), labels.to(device)
 
            optimizer.zero_grad()
            outputs = model(inputs)
 
            loss = criterion(outputs, labels)
             
            #print("Inputs OpenPrice:", labels[0, 0].item(), "Inputs ClosePrice:", labels[0, 1].item())
            #print("Predict OpenPrice:", outputs[0, 0].item(), "Predict ClosePrice:", outputs[0, 1].item())
            loss.backward() 
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f"Epoch {epoch} - Average Loss: {average_loss:.4f}")

    # # 預測並獲取預測值
    model.eval()
    torch.save(model.state_dict(), './LSTM.pth')
