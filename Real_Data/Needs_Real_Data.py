import pandas as pd
import logging as log
import sys
import numpy as np
import random
from matplotlib import pyplot as plt
log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

log.info('Loading data into pandas dataframe...')
data = pd.read_csv("Wellbeing_and_lifestyle_data_Kaggle.csv")
print(data.head(5))

log.info('Droping unnecesarry columns...') 
# Timestamp
# Age (in future if age will be available in simulator it could be used - mapped into groups)
# Gender (in future if gender will be available in simulator it could be used - binary mapping female 0, male 1)
# WORK_LIFE_BALANCE_SCORE (score will be calculated for each group, not all needs)
data = data.drop(columns=['Timestamp', 'AGE', 'GENDER', 'WORK_LIFE_BALANCE_SCORE'])

log.info('Checking mising values')
if np.any(pd.notnull(data)) == 'False':
    log.warning('Add missing value removal!!!')
else:
    print(data.info())

log.info('Checking type of values')
for column in data.columns:
    if data[column].dtype != np.int64:
        data[column] = data[column].astype('int64')
        log.info(f'Column {column} type changed to INT...')


log.info('Normalization of data to [0-1]...')
#from 10 to 0
#SUPPORTING_OTHERS, ACHIEVEMENT, PERSONAL_AWARDS, FLOW, SOCIAL_NETWORK, DAILY_STEPS, SLEEP_HOURS, TODO_COMPLETED, 
#LIVE_VISION, TIME_FOR_PASSION, PLACES_VISITED, WEEKLY_MEDITATION, CORE_CIRCLE
data['SUPPORTING_OTHERS'] = data['SUPPORTING_OTHERS'] / 10
data['ACHIEVEMENT'] = data['ACHIEVEMENT'] / 10
data['PERSONAL_AWARDS'] = data['PERSONAL_AWARDS'] / 10
data['FLOW'] = data['FLOW'] / 10
data['SOCIAL_NETWORK'] = data['SOCIAL_NETWORK'] / 10
data['DAILY_STEPS'] = data['DAILY_STEPS'] / 10
data['SLEEP_HOURS'] = data['SLEEP_HOURS'] / 10
data['TODO_COMPLETED'] = data['TODO_COMPLETED'] / 10
data['LIVE_VISION'] = data['LIVE_VISION'] / 10
data['TIME_FOR_PASSION'] = data['TIME_FOR_PASSION'] / 10
data['PLACES_VISITED'] = data['PLACES_VISITED'] / 10
data['WEEKLY_MEDITATION'] = data['WEEKLY_MEDITATION'] / 10
data['CORE_CIRCLE'] = data['CORE_CIRCLE'] / 10
#from 10 to 0 - reverse
#LOST_VACATION, DAILY_SHOUTING
data['LOST_VACATION'] = 1 - (data['LOST_VACATION'] / 10)
data['DAILY_SHOUTING'] = 1 - (data['DAILY_SHOUTING'] / 10)
#from 5 to 0
#DONATION, FRUITS_VEGGIES
data['DONATION'] = data['DONATION'] / 5
data['FRUITS_VEGGIES'] = data['FRUITS_VEGGIES'] / 5
#from 5 to 0 - reverse
#DAILY_STRESS
data['DAILY_STRESS'] = 1 - (data['DAILY_STRESS'] / 5)
#from 1 to 2
#BMI_RANGE, SUFFICIENT_INCOME
data['BMI_RANGE'] = data['BMI_RANGE'] - 1
data['SUFFICIENT_INCOME'] = data['SUFFICIENT_INCOME'] - 1

#1.SOCIAL_NETWORK, LOST_VACATION, DONATION, DAILY_STRESS, TIME_FOR_PASSION, DAILY_SHOUTING, CORE_CIRCLE, WEEKLY_MEDITATION, PLACES_VISITED, SUFFICIENT_INCOME
#2.SUPPORTING_OTHERS, FLOW, FRUITS_VEGGIES, BMI_RANGE, SLEEP_HOURS, ACHIEVEMENT, PERSONAL_AWARDS, DAILY_STEPS, TODO_COMPLETED, LIVE_VISION  
# 
# Low           5:FRUITS_VEGGIES    4:SLEEP_HOURS       3:BMI_RANGE         2:DAILY_STEPS       1:TODO_COMPLETED
# Healthy_body  5:BMI_RANGE         4:FRUITS_VEGGIES    3:DAILY_STEPS       2:SLEEP_HOURS       1:FLOW
# Healthy_mind  5:DAILY_STRESS      4:FLOW              3:LOST_VACATION     2:TIME_FOR_PASSION  1:WEEKLY_MEDITATION
# Knowl&achiev  5:ACHIEVEMENT       4:PERSONAL_AWARDS   3:PLACES_VISITED    2:SUFFICIENT_INCOME 1:LIVE_VISION
# Social_life   5:SOCIAL_NETWORK    4:CORE_CIRCLE       3:SUPPORTING_OTHERS 2:DONATION          1:DAILY_SHOUTING
# Dreams        5:LIVE_VISION       4:TODO_COMPLETED    3:PERSONAL_AWARDS   2:ACHIEVEMENT       1:SUPPORTING_OTHERS
# Desired output format
# 6 x needs, 1 x decison

log.info('Averaging all values into 5 groups...')
data_out_columns = ['Low','Healthy_body','Healthy_mind','Knowl&achiev','Social_life','Dreams','Decison']
data_out = pd.DataFrame(index=data.index, columns=data_out_columns)
data_out['Low'] = data.apply(lambda row: (((row['FRUITS_VEGGIES'] * 5) + (row['SLEEP_HOURS'] * 4) + (row['BMI_RANGE'] * 3) + \
                                           (row['DAILY_STEPS'] * 2) + (row['TODO_COMPLETED'] * 1)) / 15), axis=1)
data_out['Healthy_body'] = data.apply(lambda row: (((row['BMI_RANGE'] * 5) + (row['FRUITS_VEGGIES'] * 4) + (row['DAILY_STEPS'] * 3) + \
                                                    (row['SLEEP_HOURS'] * 2) + (row['FLOW'] * 1)) / 15), axis=1)
data_out['Healthy_mind'] = data.apply(lambda row: (((row['DAILY_STRESS'] * 5) + (row['FLOW'] * 4) + (row['LOST_VACATION'] * 3) + \
                                                    (row['TIME_FOR_PASSION'] * 2) + (row['WEEKLY_MEDITATION'] * 1)) / 15), axis=1)
data_out['Knowl&achiev'] = data.apply(lambda row: (((row['ACHIEVEMENT'] * 5) + (row['PERSONAL_AWARDS'] * 4) + (row['PLACES_VISITED'] * 3) + \
                                                    (row['SUFFICIENT_INCOME'] * 2) + (row['LIVE_VISION'] * 1)) / 15), axis=1)
data_out['Social_life'] = data.apply(lambda row: (((row['SOCIAL_NETWORK'] * 5) + (row['CORE_CIRCLE'] * 4) + (row['SUPPORTING_OTHERS'] * 3) + \
                                                    (row['DONATION'] * 2) + (row['DAILY_SHOUTING'] * 1)) / 15), axis=1)
data_out['Dreams'] = data.apply(lambda row: (((row['LIVE_VISION'] * 5) + (row['TODO_COMPLETED'] * 4) + (row['PERSONAL_AWARDS'] * 3) + \
                                              (row['ACHIEVEMENT'] * 2) + (row['SUPPORTING_OTHERS'] * 1)) / 15), axis=1)

log.info('Normalazing again the data to extend to whole range 0-1...')
data_out['Low']=(data_out['Low']-data_out['Low'].min())/(data_out['Low'].max()-data_out['Low'].min())
data_out['Healthy_body']=(data_out['Healthy_body']-data_out['Healthy_body'].min())/(data_out['Healthy_body'].max()-data_out['Healthy_body'].min())
data_out['Healthy_mind']=(data_out['Healthy_mind']-data_out['Healthy_mind'].min())/(data_out['Healthy_mind'].max()-data_out['Healthy_mind'].min())
data_out['Knowl&achiev']=(data_out['Knowl&achiev']-data_out['Knowl&achiev'].min())/(data_out['Knowl&achiev'].max()-data_out['Knowl&achiev'].min())
data_out['Social_life']=(data_out['Social_life']-data_out['Social_life'].min())/(data_out['Social_life'].max()-data_out['Social_life'].min())
data_out['Dreams']=(data_out['Dreams']-data_out['Dreams'].min())/(data_out['Dreams'].max()-data_out['Dreams'].min())
#data_out.hist(bins=10)
#plt.show()

log.info('Decision making process...')
def decision_making (row):
    if row['Low'] < random.uniform(0.5, 0.65) :
        return 'a_go_low'
    if row['Healthy_body'] < random.uniform(0.4, 0.6) :
        return 'b_go_healthy_body'
    if row['Healthy_mind'] < random.uniform(0.4, 0.6) :
        return 'c_go_healthy_mind'
    if row['Knowl&achiev'] < random.uniform(0.4, 0.6) :
        return 'd_knowl&achiev'
    if row['Social_life'] < random.uniform(0.4, 0.6) :
        return 'e_social_life'
    if row['Dreams'] < random.uniform(0.4, 0.6) :     
        return 'f_dreams'
    else:
        all_needs = ['a_go_low','b_go_healthy_body','c_go_healthy_mind','d_knowl&achiev','e_social_life','f_dreams']
        return random.choice(all_needs)

data_out['Decison'] = data_out.apply(lambda row: decision_making (row), axis=1)

data_out.to_csv('real_needs.csv', index=False)
log.info("End - output file real_needs.csv")