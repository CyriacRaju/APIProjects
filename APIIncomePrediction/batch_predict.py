import requests
import pandas as pd
import sys


# Get the files
input_csv = sys.argv[1]
output_csv = sys.argv[2]

# Url
url = "http://127.0.0.1:5000/predict"

# Loading input.csv
batch_data = pd.read_csv(input_csv)
batch_data = batch_data.to_dict(orient='records')

# Getting predictions
response_list = []
for row in batch_data:
    response = requests.post(url, json=row)
    response_list.append(dict(response.json()))

# Saving output.csv
response_df = pd.DataFrame(response_list)
response_df.to_csv(output_csv, index=False)

