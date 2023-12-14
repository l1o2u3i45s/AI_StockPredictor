import matplotlib.pyplot as plt
import json

stock_id = '006208'
with open('Data/' + stock_id + '.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

    close_prices = [] 
    date = []
    idx = 0
    for record in json_data:
        close_prices.append(record['ClosePrice'])
        date.append(idx)
        idx += 1
 
    plt.plot(date,close_prices)
    plt.title(f'Close Prices for {stock_id}')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.show()
