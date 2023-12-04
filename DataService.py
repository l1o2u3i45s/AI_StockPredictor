
import json
import torch
stock_id = "006208"

def GetDModel():
    return 9

def GetData():
# Reading JSON data (replace with the path to your JSON file)
    with open('Data/' + stock_id + '.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file) 

    tensors = []
    labels = []
    for record in json_data:
        jsonData = [record['OpenPrice'],
               record['MaxPrice'],record['MinPrice'],
               record['ClosePrice'],record['MA5'],
               record['MA10'],record['MA20'],
               record['MA60'],record['MACDSignal']  ]
        labelData = [1 if record['OpenPrice'] < record['ClosePrice'] else 0,
                     0 if record['OpenPrice'] < record['ClosePrice'] else 1] 

        tensorData = torch.tensor(jsonData, dtype=torch.float32)
        label = torch.tensor(labelData, dtype=torch.float32)
        tensors.append(tensorData)
        labels.append(label)

    return tensors, labels