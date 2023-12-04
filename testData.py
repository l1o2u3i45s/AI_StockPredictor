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

tensors,labels = DataService.GetData()

trainDatasize = int(len(tensors) * 0.8)
 
testTensors = tensors[trainDatasize+1 : len(tensors)] 
testLabelTensor = labels[trainDatasize+1 : len(labels)] 
# 轉換為PyTorch張量
window_size = 23
maskTensor = torch.tensor([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])

 
 
testData = [torch.stack(testTensors[i:i+window_size]) for i in range(len(testTensors) - window_size)]
testMaskData = [torch.stack(testTensors[i:i+window_size] + [maskTensor]) for i in range(len(testTensors) - window_size)]
testLabel = testLabelTensor[window_size : len(testLabelTensor)]
 
 
testDataSet = Model.ModelDataset(testData,testMaskData, testLabel)


# # 創建數據加載器  
test_loader = DataLoader(testDataSet, batch_size=1)

test_losses = []


# # 實例化模型、損失函數和優化器
trainType = 2
if trainType == 1:

    testModel = Model.Transformer(input_dim= 10) 
    testModel.load_state_dict(torch.load('./TransFormer.pth'))

    criterion = nn.MSELoss()
    
    # No need to track gradients for evaluation
    with torch.no_grad():
        for inputs, inputMask, labels in test_loader:
            
            
            inputs, inputMask, labels = inputs , inputMask , labels 

            outputs = testModel(inputs, inputMask)

            print("label OpenPrice:", labels[0, 0].item(), "label ClosePrice:", labels[0, 1].item())
            print("Predict OpenPrice:", outputs[0, 0].item(), "Predict ClosePrice:", outputs[0, 1].item())

            loss = criterion(outputs, labels)
            test_losses.append(loss.item())

    # Calculate the average loss over all test batches
    average_test_loss = np.mean(test_losses)
    print(f"Average test loss: {average_test_loss}")

elif trainType == 2:
    testModel = Model.LSTM(dimension = 10).to(device) 
    testModel.load_state_dict(torch.load('./LSTM.pth'))

    criterion = nn.MSELoss()
    count = 0
    # No need to track gradients for evaluation
    totalScore = 0
    correctCnt = 0
    with torch.no_grad():
        for inputs, inputMask, labels in test_loader:
            print(count)
            count +=1
            inputs, inputMask, labels = inputs.to(device), inputMask.to(device), labels.to(device)

            outputs = testModel(inputs)

            predict = 0
            if(outputs[0] >= 0.5):
                predict = 1

            if(predict == labels[0]):
                correctCnt +=1

            totalScore +=1

            #print("label OpenPrice:", labels[0, 0].item(), "label ClosePrice:", labels[0, 1].item())
            #print("Predict OpenPrice:", outputs[0, 0].item(), "Predict ClosePrice:", outputs[0, 1].item())

            loss = criterion(outputs, labels)
            test_losses.append(loss.item())

    # Calculate the average loss over all test batches
    average_test_loss = np.mean(test_losses)
    print(f"Average test loss: {correctCnt/totalScore * 100}")