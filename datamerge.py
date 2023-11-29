import json

code = '006208'
# Reading the JSON data from the files
with open( code + '_stock.txt', 'r', encoding='utf-8') as file:
    stock_json = json.load(file)

with open(code + '_BuySell.txt', 'r', encoding='utf-8') as file:
    buysell_json = json.load(file)

# Extracting the 'data' part from both JSON objects
stock_data = stock_json['data']
buysell_data = buysell_json['data']

# Creating a dictionary to hold the merged data
merged_data = {}

# Merging stock data
for entry in stock_data:
    date = entry['date']
    if date not in merged_data:
        merged_data[date] = {key: entry[key] for key in entry if key != 'date'}
    else:
        for key in entry:
            if key != 'date':
                merged_data[date][key] = entry[key]

# Merging buy/sell data
for entry in buysell_data:
    date = entry['date']
    name = entry['name']
    if date in merged_data:
        if name not in merged_data[date]:
            merged_data[date][name] = {'buy': 0, 'sell': 0}
        merged_data[date][name]['buy'] += entry['buy']
        merged_data[date][name]['sell'] += entry['sell']

# Modifying the JSON structure
modified_merged_data = {'data': []}

for date, data in merged_data.items():
    entry = {'date': date}
    for key, value in data.items():
        if isinstance(value, dict):
            entry[key] = value
        else:
            entry[key] = value
    modified_merged_data['data'].append(entry)

# Writing the modified merged data to a JSON file
output_json_path = 'path_to_output_json.json'
with open(output_json_path, 'w', encoding='utf-8') as file:
    json.dump(modified_merged_data, file, ensure_ascii=False, indent=4)
