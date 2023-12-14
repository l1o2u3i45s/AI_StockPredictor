import torch  
from torch.utils.data import DataLoader 
import Model
import DataService

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tensors,labels = DataService.GetData()
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
test_loader = DataLoader(testDataSet,shuffle=True, batch_size=1)
 
 
# # 實例化模型、損失函數和優化器
trainType = 2
if trainType == 1:

    testModel = Model.Transformer(input_dim= input_DModel) 
    testModel.load_state_dict(torch.load('./TransFormer.pth'))
 
    totalScore = 0
    correctCnt = 0
 
    with torch.no_grad():
        for inputs, inputMask, labels in test_loader:
            inputs, inputMask, labels = inputs , inputMask , labels 

            outputs = testModel(inputs, inputMask)

            predict = 0
            if(outputs[0,0] >= 0.5):
                predict = 1

            if(predict == labels[0,0]):
                correctCnt +=1

            totalScore +=1

    print(f"Correct Rate: {correctCnt/totalScore * 100}")

elif trainType == 2:
    testModel = Model.LSTM(dimension = input_DModel).to(device) 
    testModel.load_state_dict(torch.load('./LSTM.pth'))
   
    totalScore = 0
    correctCnt = 0
    with torch.no_grad():
        for inputs, inputMask, labels in test_loader: 
            inputs, inputMask, labels = inputs.to(device), inputMask.to(device), labels.to(device)

            outputs = testModel(inputs)
 

            predict = 0

            if(outputs[0] >= 0.5):
                predict = 1

            if(predict == labels[0]):
                correctCnt +=1

            totalScore +=1
 
    print(f"Correct Rate: {correctCnt/totalScore * 100}")