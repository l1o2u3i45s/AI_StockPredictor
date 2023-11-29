import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import Model

stock_id = "006208"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
  
# Reading JSON data (replace with the path to your JSON file)
with open('Data/' + stock_id + '_RawData.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Convert JSON to DTO
stock_dto = Model.json_to_dto(json_data)
 
tensors = [Model.stock_data_to_tensor(data) for data in stock_dto.data]

# 轉換為PyTorch張量
window_size = 20
tensors_grouped = [torch.stack(tensors[i:i+window_size]) for i in range(len(tensors) - window_size + 1)]

print(tensors_grouped[0].shape)

 

# # 創建數據加載器
# train_data = TensorDataset(X_train, Y_train)
# test_data = TensorDataset(X_test, Y_test)
# train_loader = DataLoader(train_data, shuffle=True, batch_size=16)
# test_loader = DataLoader(test_data, batch_size=16)



# # 實例化模型、損失函數和優化器
# model = Model.StockPredictor(input_dim=5, hidden_dim=50, num_layers=2, output_dim=1).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 訓練模型
# num_epochs = 100
# for epoch in range(num_epochs):
#     print(epoch)
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels.unsqueeze(1))
#         loss.backward()
#         optimizer.step()

# # 預測並獲取預測值
# model.eval()
 
