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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

stockRawData = []

# 讀取數據
with open('006208_stock.txt') as f:
    data = json.load(f)
    stockdata = data['data']

with open('006208_BuySell.txt') as f:
    data = json.load(f)
    stockBuySelldata = data['data']

# 轉換數據為數字格式
data = []
for rawdata in stockdata:
    data.append([
        float(rawdata['Trading_Volume']),
        float(rawdata['open']),
        float(rawdata['max']),
        float(rawdata['min']),
        float(rawdata['close']),
        float(rawdata['spread']) 
    ])

# 標準化數據
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 創建特徵和標籤
def create_dataset(data, time_step):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0:-1])
        Y.append(data[i + time_step, -1])
    return np.array(X), np.array(Y)

time_step = 20
X, Y = create_dataset(scaled_data, time_step)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 轉換為PyTorch張量
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)

# 創建數據加載器
train_data = TensorDataset(X_train, Y_train)
test_data = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_data, shuffle=True, batch_size=16)
test_loader = DataLoader(test_data, batch_size=16)

# 定義LSTM模型
class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(StockPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach())).to(device)
        out = self.fc(out[:, -1, :]).to(device)
        return out

# 實例化模型、損失函數和優化器
model = StockPredictor(input_dim=5, hidden_dim=50, num_layers=2, output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 100
for epoch in range(num_epochs):
    print(epoch)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

# 預測並獲取預測值
model.eval()
with torch.no_grad():
    predicted_stock_price = model(X_test).cpu().detach().numpy()

# 確保predicted_stock_price是一個二維陣列，並且其第二維度與X_test相匹配
predicted_stock_price_2d = predicted_stock_price.reshape(-1, 1)

# 確保predicted_stock_price_2d與X_test在列上有相同的維度
# 我們需要添加足夠的列以匹配X_test的最後一個時間步的特徵數量
num_features = X_test.shape[2] - 1  # X_test中每個時間步的特徵數量，減去1是因為我們要添加預測值
additional_cols = np.zeros((predicted_stock_price_2d.shape[0], num_features - predicted_stock_price_2d.shape[1]))

# 現在合併additional_cols和predicted_stock_price_2d
predicted_stock_price_expanded = np.concatenate((additional_cols, predicted_stock_price_2d), axis=1)

# 這樣我們就可以和X_test的最後一個時間步合併了
predicted_combined = np.concatenate((X_test[:, -1, :-1].cpu().numpy(), predicted_stock_price_expanded), axis=1)
predicted_stock_price_transformed = scaler.inverse_transform(predicted_combined)[:, -1]

# 對於真實價格的處理應該保持不變
real_stock_price_2d = Y_test.cpu().numpy().reshape(-1, 1)
real_combined = np.concatenate((X_test[:, -1, :-1].cpu().numpy(), real_stock_price_2d), axis=1)
real_stock_price_transformed = scaler.inverse_transform(real_combined)[:, -1]

# 繪製預測與實際曲線圖...


# 繪製預測與實際曲線圖
plt.figure(figsize=(10,6))
plt.plot(real_stock_price_transformed, color='blue', label='Real Stock Price')
plt.plot(predicted_stock_price_transformed, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
