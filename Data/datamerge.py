
import json

code = '006208'
# Reading the JSON data from the files
with open(code + '_stock.txt', 'r', encoding='utf-8') as file:
    stock_json = json.load(file)

with open(code + '_BuySell.txt', 'r', encoding='utf-8') as file:
    buysell_json = json.load(file)

# Extracting the 'data' part from both JSON objects
stock_data = stock_json['data']
buysell_data = buysell_json['data']

# Creating a dictionary to hold the intermediate merged data
intermediate_merged_data = {}

# Merging stock data
for entry in stock_data:
    date = entry['date']
    if date not in intermediate_merged_data:
        intermediate_merged_data[date] = {key: entry[key] for key in entry if key != 'date'}
    else:
        for key in entry:
            if key != 'date':
                intermediate_merged_data[date][key] = entry[key]

# Merging buy/sell data
for entry in buysell_data:
    date = entry['date']
    name = entry['name']
    if date in intermediate_merged_data:
        if 'BuySell' not in intermediate_merged_data[date]:
            intermediate_merged_data[date]['BuySell'] = []
        if name in ['Foreign_Investor', 'Investment_Trust', 'Dealer']:
            intermediate_merged_data[date]['BuySell'].append({'name': name, 'buy': entry['buy'], 'sell': entry['sell']})

# Converting intermediate merged data to the final structure with "data" as the key
final_merged_data = {'data': [{'date': date, **data} for date, data in intermediate_merged_data.items()]}

# Writing the final merged data to a JSON file
output_json_path = code + '_RawData.json'
with open(output_json_path, 'w', encoding='utf-8') as file:
    json.dump(final_merged_data, file, ensure_ascii=False, indent=4)
