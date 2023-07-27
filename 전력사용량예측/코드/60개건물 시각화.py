#%%
import numpy as np
import math
import pandas as pd

#Visualizing
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn.palettes import color_palette
train = pd.read_csv("/Users/unixking/Desktop/python/데이터누리/energy/csv/train.csv", encoding="euc-kr")
data =  pd.read_csv("/Users/unixking/Desktop/python/데이터누리/energy/csv/test33.csv", encoding="euc-kr")



fig = plt.figure(figsize=(25,25)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정

for i in range(1,61):
    train2 = train[train.num == i]
    data2 = data[data.num == i]

    train2  = train2[["전력사용량(kWh)"]]
    test2 = data2[["전력사용량"]]

    train2 = train2[1872:]
    test2 = test2[:168]


    train2= train2.reset_index()
    test2 = test2.reset_index()

    ax = fig.add_subplot(8,8,i) ## 그림 뼈대(프레임) 생성

    plt.xlabel("num_{}".format(i)) 
    plt.ylabel('평균전력사용량')

    ax.plot(train2["전력사용량(kWh)"],marker='',label='train', color="blue")
    ax.plot(test2["전력사용량"], label=test2, color="red") 
plt.show()
# %%
