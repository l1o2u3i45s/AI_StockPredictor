import torch  
from torch.utils.data import DataLoader 
import Model
import DataService 
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tensors,labels = DataService.GetPriceData()
input_DModel = DataService.GetDModel()
trainDatasize = int(len(tensors) * 0.8)
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
outputs_list = []  # 用於存儲模型的輸出
labels_list = []   # 用於存儲標籤 
 
# # 實例化模型、損失函數和優化器
trainType = 1
if trainType == 1:

    testModel = Model.Transformer_Price(input_dim= input_DModel) 
    testModel.load_state_dict(torch.load('./TransFormer_Price.pth'))
   
    with torch.no_grad():
        for inputs, inputMask, labels in test_loader:
            inputs, inputMask, labels = inputs , inputMask , labels 

            outputs = testModel(inputs, inputMask)

            outputs_list.extend(outputs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

 

elif trainType == 2:
    testModel = Model.LSTM_Price(dimension = input_DModel).to(device) 
    testModel.load_state_dict(torch.load('./LSTM_Price.pth'))
   

    with torch.no_grad():
        for inputs, inputMask, labels in test_loader: 
            inputs, inputMask, labels = inputs.to(device), inputMask.to(device), labels.to(device)

            outputs = testModel(inputs)
 
            outputs_list.extend(outputs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
 
 # 現在使用 Matplotlib 繪製折線圖
plt.figure(figsize=(10, 6))
plt.plot(outputs_list, label='Model Outputs')
plt.plot(labels_list, label='Labels')
plt.title('Comparison between Model Outputs and Labels')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.show()
 