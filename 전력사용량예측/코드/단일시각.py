#%%
import numpy as np
import math
import pandas as pd
from pandas.core.indexes import interval


#Visualizing
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn.palettes import color_palette
from matplotlib import dates

plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False;

train = pd.read_csv("/Users/unixking/Desktop/python/데이터누리/energy/csv/train.csv", encoding="euc-kr")
data =  pd.read_csv("/Users/unixking/Desktop/python/데이터누리/energy/csv/test33.csv", encoding="euc-kr")

fig = plt.figure(figsize=(25,25)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정

i = 2
train2 = train[train.num == i]
data2 = data[data.num == i]

train2  = train2[["전력사용량(kWh)", 'date_time']]
test2 = data2[["전력사용량", 'date_time']]

train2 = train2[1536:]

#test2 = test2[:168]


train2= train2.reset_index()
test2 = test2.reset_index()


fig = plt.figure(figsize=(18,18)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정
ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성

plt.title("MLP & LSTM")
plt.xlabel("date_time") 
plt.ylabel('전력사용량')
plt.xticks(rotation=45)
ax.xaxis.set_major_locator(dates.MonthLocator(interval = 2))

ax.plot(train2['date_time'], train2["전력사용량(kWh)"], marker='',label= '8월03일부터 8월24일', color="blue",linewidth=2.5)
ax.plot(test2['date_time'], test2["전력사용량"], marker='', label= "8월25일부터 8월31일", color="red",linewidth=2.5) 
plt.show()
# %%
