import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import Model

stock_id = "006208"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
  
# Reading JSON data (replace with the path to your JSON file)
with open('Data/' + stock_id + '.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file) 

tensors = []
labels = []
for record in json_data:
    jsonData = [record['TradeVolumn'],record['OpenPrice'],
               record['MaxPrice'],record['MinPrice'],
               record['ClosePrice'],record['MA5'],
               record['MA10'],record['MA20'],
               record['MA60'],record['MACDSignal']  ]
    labelData = [record['OpenPrice'],record['ClosePrice']] 

    tensorData = torch.tensor(jsonData, dtype=torch.float32) 
    label = torch.tensor(labelData, dtype=torch.float32) 
    tensors.append(tensorData)
    labels.append(label)

trainDatasize = int(len(tensors) * 0.8)
 

trainTensors = tensors[0 : trainDatasize] 
trainLabelTensor = labels[0 : trainDatasize] 
testTensors = tensors[trainDatasize+1 : len(tensors)] 
testLabelTensor = labels[trainDatasize+1 : len(labels)] 
# 轉換為PyTorch張量
window_size = 100
maskTensor = torch.tensor([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])

trainData = [torch.stack(trainTensors[i:i+window_size]) for i in range(len(trainTensors) - window_size)]
trainMaskData = [torch.stack(trainTensors[i:i+window_size]+ [maskTensor]) for i in range(len(trainTensors) - window_size)]
trainLabel = trainLabelTensor[window_size : len(trainLabelTensor)]
 
testData = [torch.stack(testTensors[i:i+window_size]) for i in range(len(testTensors) - window_size)]
testMaskData = [torch.stack(testTensors[i:i+window_size] + [maskTensor]) for i in range(len(testTensors) - window_size)]
testLabel = testLabelTensor[window_size : len(testLabelTensor)]
 

trainDataSet = Model.ModelDataset(trainData,trainMaskData, trainLabel)
testDataSet = Model.ModelDataset(testData,testMaskData, testLabel)


# # 創建數據加載器 

test_loader = DataLoader(testDataSet, batch_size=16)



# # 實例化模型、損失函數和優化器
trainType = 1

if trainType == 1: #TransFormer
    train_loader = DataLoader(trainDataSet, shuffle=True, batch_size=16)
    model = Model.Transformer(input_dim= 10) 
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
            inputs,inputMask, labels = inputs  ,inputMask  , labels 
 
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
    model = Model.LSTM(dimension = 10).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # 訓練模型
    num_epochs = 300
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
