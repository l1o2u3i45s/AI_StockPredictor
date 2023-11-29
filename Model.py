from typing import List, Dict
import torch.nn as nn 
import torch 

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
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach())) 
        out = self.fc(out[:, -1, :]) 
        return out


class BuySellDTO:
    def __init__(self, buy: int, sell: int):
        self.buy = buy
        self.sell = sell

class StockDataDTO:
    def __init__(self, date: str, stock_id: str, Trading_Volume: int, Trading_money: int, 
                 open: float, max: float, min: float, close: float, spread: float, 
                 Trading_turnover: int, buy_sell_data: Dict[str, BuySellDTO]):
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

# Function to convert JSON data to DTO
def json_to_dto(json_data: Dict) -> StockDTO:
    stock_list = []
    for entry in json_data['data']:
        buy_sell_data = {k: BuySellDTO(v['buy'], v['sell']) for k, v in entry.items() if isinstance(v, dict)}
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


