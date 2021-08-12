import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
from datetime import datetime
import os

directory = "C:/PBL/PBL_DecisionModel/Simulations"
filetype  = "*.csv"

files = [f for f in os.listdir(directory) if f.endswith(filetype[1:])]
for file in files:
    data = pd.read_csv(file, error_bad_lines=False, header=None)
    file_split = file.split('_')
    for itt, row in data.iterrows():
        if data.at[itt,0] == '#TIMESTAMP':
            date_time_obj = data.at[itt+1,0]
        if data.at[itt,0] == '#INFECTED':
            if data.at[itt+1,0] == file_split[1]:
                break               
    print(f'File: {file} - Time: {date_time_obj}.')