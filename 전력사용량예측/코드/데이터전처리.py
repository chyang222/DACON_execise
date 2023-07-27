import numpy as np
import pandas as pd
from pandas.core import groupby
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

df = pd.read_csv("C:\\Users\\user\\Desktop\\energy\\test.csv", encoding="cp949")

train = pd.read_csv("/Users/unixking/Desktop/c언어 실습/데이터누리/energy/csv/exercise.csv", encoding="euc-kr")
test =  pd.read_csv("/Users/unixking/Desktop/c언어 실습/데이터누리/energy/csv/e.csv", encoding="euc-kr")

tem = df['기온']
df['기온'] = pd.Series(tem).interpolate()

wi = df['풍속']
df['풍속'] = pd.Series(wi).interpolate()

wa = df['습도']
df['습도'] = pd.Series(wa).interpolate()

ra = df['강수량(mm, 6시간)']
df['강수량(mm, 6시간)'] = pd.Series(ra).interpolate()

su = df['일조(hr, 3시간)']
df['일조(hr, 3시간)'] = pd.Series(su).interpolate()

a = df[df['비전기냉방설비운영'] == 1.0]['num']
b = a.drop_duplicates()

for i in b:
    df.loc[df['num']==i, '비전기냉방설비운영'] =1.0

c = df[df['태양광보유'] == 1.0]['num']
d = a.drop_duplicates()

for i in d:
    df.loc[df['num']==i, '태양광보유'] =1.0

df.to_csv("C:\\Users\\user\\Desktop\\energy\\test.csv", index=False, encoding='cp949') 

def get_pow(series):
    return math.pow(series, 0.15)


train['체감온도'] = 13.12 + 0.6215*train['기온']-11.37*train['풍속(m/s)'].apply(get_pow) + 0.3965*train['풍속(m/s)'].apply(get_pow)*train['기온']
train.to_csv("/Users/unixking/Desktop/c언어 실습/데이터누리/energy/csv/exercise2.csv", index=False, encoding='cp949') 