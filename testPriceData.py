import torch  
from torch.utils.data import DataLoader 
import Model
import DataService 
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tensors,labels = DataService.GetPriceData()
input_DModel = DataService.GetDModel()
trainDatasize = int(len(tensors) * 0.7)
print(f"datasize:{trainDatasize}")

testTensors = tensors[trainDatasize+1 : len(tensors)] 
testLabelTensor = labels[trainDatasize+1 : len(labels)] 
# 轉換為PyTorch張量
window_size = DataService.GetWindowSize()
maskTensor = DataService.GetMaskData()
 
testData = [torch.stack(testTensors[i:i+window_size]) for i in range(len(testTensors) - window_size)]
testMaskData = [torch.stack(testTensors[i:i+window_size] + [maskTensor]) for i in range(len(testTensors) - window_size)]
testLabel = testLabelTensor[window_size : len(testLabelTensor)]
  
testDataSet = Model.ModelDataset(testData,testMaskData, testLabel)


# # 創建數據加載器  
test_loader = DataLoader(testDataSet, batch_size=1)
outputs_list_open = []  # 用於存儲模型的輸出
labels_list_open = []   # 用於存儲標籤 
outputs_list_close = []  # 用於存儲模型的輸出
labels_list_close = []   # 用於存儲標籤 
 
# # 實例化模型、損失函數和優化器
trainType = 2
if trainType == 1:

    testModel = Model.Transformer_Price(input_dim= input_DModel) 
    testModel.load_state_dict(torch.load('./TransFormer_Price.pth'))
   
    with torch.no_grad():
        for inputs, inputMask, labels in test_loader:
            inputs, inputMask, labels = inputs , inputMask , labels 

            outputs = testModel(inputs, inputMask)
 

 

elif trainType == 2:
    testModel = Model.LSTM_Price(dimension = input_DModel).to(device) 
    testModel.load_state_dict(torch.load('./LSTM_Price.pth'))
    criterion = nn.MSELoss()
    total_loss = 0
    num_batches = 0

    correctCnt = 0
    totalCnt = 0
    totalEaringPrice = 0
    with torch.no_grad():
        for inputs, inputMask, labels in test_loader: 
            inputs, inputMask, labels = inputs.to(device), inputMask.to(device), labels.to(device)

            outputs = testModel(inputs)

            outputs_array = np.array([outputs.cpu()])
 
            outputs_list_open.append(outputs_array[0,0][0])
            labels_list_open.append(labels[0][0].cpu().numpy())
            outputs_list_close.append(outputs_array[0,0][1])
            labels_list_close.append(labels[0][1].cpu().numpy())

            predictUp = outputs_array[0,0][1] > outputs_array[0,0][0]
            labelUp = labels[0][1] > labels[0][0]

            if predictUp == labelUp :
                correctCnt +=1

            totalCnt +=1

            if predictUp == True:
                earingPrice = outputs_array[0,0][0] - outputs_array[0,0][1]
                print(f"earingPrice:{earingPrice}")
                totalEaringPrice += earingPrice

            loss = criterion(outputs, labels)
 
            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f"Average Loss: {average_loss:.4f}")
        print(f"Correct Rate: {correctCnt / totalCnt * 100} %")
        print(f"Total Earing:{totalEaringPrice}")
 
 # 現在使用 Matplotlib 繪製折線圖
plt.figure(figsize=(10, 6))
plt.plot(outputs_list_close, label='Model Outputs')
plt.plot(labels_list_close, label='Labels')
plt.title('收盤價預測')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.show()
plt.close() 

plt.figure(figsize=(10, 6))
plt.plot(outputs_list_open, label='Model Outputs')
plt.plot(labels_list_open, label='Labels')
plt.title('開盤價預測')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.show()