from typing import List, Dict
import torch.nn as nn 
import torch 
from torch.utils.data import Dataset


# 定義模型
class StockPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StockPredictor, self).__init__()
        
        d_model = input_dim
        nhead = input_dim // 4
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

# Re-defining the DTO classes
class BuySellDTO:
    def __init__(self, buy: int, sell: int):
        self.buy = buy
        self.sell = sell

class StockDataDTO:
    def __init__(self, date: str, stock_id: str, Trading_Volume: int, Trading_money: int, 
                 open: float, max: float, min: float, close: float, spread: float, 
                 Trading_turnover: int, buy_sell_data: List[BuySellDTO]):
        self.date = date
        self.stock_id = stock_id
        self.Trading_Volume = Trading_Volume
        self.Trading_money = Trading_money
        self.open = open
        self.max = max
        self.min = min
        self.close = close
        self.spread = spread
        self.Trading_turnover = Trading_turnover
        self.buy_sell_data = buy_sell_data

class StockDTO:
    def __init__(self, data: List[StockDataDTO]):
        self.data = data

# Function to convert JSON data to DTO, merging Foreign_Investor, Investment_Trust, Dealer into BuySell
def json_to_dto(json_data: Dict) -> StockDTO:
    stock_list = []
    for entry in json_data:
        buy_sell_data = []
        buy_sell_data.append(BuySellDTO(entry['BuySell'][0]['buy'], entry['BuySell'][0]['sell']))
        buy_sell_data.append(BuySellDTO(entry['BuySell'][1]['buy'], entry['BuySell'][1]['sell'])) 
        stock_data_dto = StockDataDTO(
            date=entry['date'], 
            stock_id=entry['stock_id'], 
            Trading_Volume=entry['Trading_Volume'], 
            Trading_money=entry['Trading_money'], 
            open=entry['open'], 
            max=entry['max'], 
            min=entry['min'], 
            close=entry['close'], 
            spread=entry['spread'], 
            Trading_turnover=entry['Trading_turnover'], 
            buy_sell_data=buy_sell_data
        )
        stock_list.append(stock_data_dto)
    return StockDTO(stock_list)

 


# Dummy function to convert StockDataDTO to a tensor
def stock_data_to_tensor(stock_data: StockDataDTO) -> torch.Tensor:
    values = [ 
        stock_data.open,
        stock_data.max,
        stock_data.min,
        stock_data.close,
        stock_data.buy_sell_data[0].buy,
        stock_data.buy_sell_data[0].sell,
        stock_data.buy_sell_data[1].buy,
        stock_data.buy_sell_data[1].sell, 
    ]
    return torch.tensor(values, dtype=torch.float32)