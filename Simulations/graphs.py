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
    
    fig,ax=plt.subplots(figsize=(16.0, 10.0))
    ax.plot(time_stamp, infected, label='Infected agents')
    ax.plot(time_stamp, infected_no_symptoms, label='Infected agents without symptoms')
    ax.plot(time_stamp, infected_mild_symptoms, label='Infected agents with symptoms')
    ax.plot(time_stamp, infected_seriously_ill, label='Agents seriously ill')
    ax.set_xlabel("Time [hh:mm]", fontsize=16)
    ax.set_ylabel("Number of agents", fontsize=16)
    ax.legend(frameon=False, loc='upper center', ncol=4, fontsize=13)
    xfmt = md.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    plt.gcf().autofmt_xdate()
    plt.savefig(f'{file}.png')