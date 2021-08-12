import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
from datetime import datetime
import os

directory = "C:\PBL\PBL_DecisionModel\Simulations"
filetype  = "*.csv"

files = [f for f in os.listdir(directory) if f.endswith(filetype[1:])]
for file in files:
    data = pd.read_csv(file, error_bad_lines=False, header=None)
    data.to_csv(file, index=False)
    file_split = file.split('_')
    time_stamp = []
    infected = []
    infected_no_symptoms = []
    infected_mild_symptoms = []
    infected_seriously_ill = []
    break_flag = False
    for itt, row in data.iterrows():
        if data.at[itt,0] == '#TIMESTAMP':
            date_time_obj = data.at[itt+1,0].split(' ')[-1]
            date_time_obj = datetime.strptime(date_time_obj, '%H:%M:%S')
            time_stamp.append(date_time_obj)
        if data.at[itt,0] == '#INFECTED':
            infected.append(int(data.at[itt+1,0]))
            if data.at[itt+1,0] == file_split[1]:
                break_flag = True   
        if data.at[itt,0] == '#INFECTED_NO_SYMPTOMS':
            infected_no_symptoms.append(int(data.at[itt+1,0]))   
        if data.at[itt,0] == '#INFECTED_MILD_SYMPTOMS':
            infected_mild_symptoms.append(int(data.at[itt+1,0]))   
        if data.at[itt,0] == '#INFECTED_SERIOUSLY_ILL':
            infected_seriously_ill.append(int(data.at[itt+1,0]))   
            if break_flag == True:
                break                              
    if 'HIGH' in file:
        infected_high = infected
        time_stamp_high = time_stamp
    if 'RAND' in file:
        infected_rand = infected
        time_stamp_rand = time_stamp
    if 'LOW' in file:
        infected_low = infected
        time_stamp_low = time_stamp

fig,ax=plt.subplots(figsize=(16.0, 10.0))
ax.plot(infected_low, time_stamp_low, label='Artificial model')
ax.plot(infected_high, time_stamp_high, label='Real model')
ax.plot(infected_rand, time_stamp_rand, label='Random behaviour')
ax.set_ylabel("Time [hh:mm]", fontsize=16)
ax.set_xlabel("Number of agents", fontsize=16)
ax.legend(frameon=False, loc='upper center', ncol=4, fontsize=16)
xfmt = md.DateFormatter('%H:%M')
ax.yaxis.set_major_formatter(xfmt)
plt.gcf().autofmt_xdate()
plt.savefig(f'{file}.png')