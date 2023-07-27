#%%
from keras.layers.core import Dropout
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras.backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler

train = pd.read_csv('C:\\Users\\user\\OneDrive\\바탕 화면\\energy\\exercise3.csv', encoding="euc-kr")
test = pd.read_csv('C:\\Users\\user\\OneDrive\\바탕 화면\\energy\\test33.csv', encoding="euc-kr")
corr = pd.read_csv('C:\\Users\\user\\OneDrive\\바탕 화면\\corr.csv', encoding="euc-kr")


def SMAPE(true,predicted):
    epsilon = 0.1
    summ = K.maximum(K.abs(true) + K.abs(predicted) + epsilon, 0.5 + epsilon)
    smape = K.abs(predicted - true) / summ * 2.0
    return smape

def co (corr):
    di = {}
    li = []
    for j in range(2,13):
        if abs(corr[corr.columns[j]][i-1]) > 0.3:
            li.append(corr.columns[j])
            di["num{}".format(i)] = li
    
    if len(di) <= 2:
        di["num{}".format(i)] = ['기온', '불쾌지수', '체감온도', "hour", "weekend"]
    else:
        pass

    return di


def num(train, test , i):
    test = test[test.num == i]
    train = train[train.num == i]

    train = train.reset_index()
    val_train = train[1872:2040]
    train = train[1536:1872]
    val_train = val_train.reset_index()
    
    return train, test, val_train


def wjdrb(train, test, val_train, corli, lefe):

    train = train[['전력사용량(kWh)']+ corli]
    test = test[corli]
    val_train = val_train[['전력사용량(kWh)']+ corli]


    mini=train.iloc[:,-(lefe+1)].min()
    size=train.iloc[:,-(lefe+1)].max()-train.iloc[:,-6].min()
    train.iloc[:,-(lefe+1)]=(train.iloc[:,-(lefe+1)]-mini)/size
    val_train.iloc[:,-(lefe+1)] = (val_train.iloc[:,-(lefe+1)]-mini)/size


    mini2=train.iloc[:,-lefe:].min()
    size2=train.iloc[:,-lefe:].max()-train.iloc[:,-lefe:].min()
    train.iloc[:,-lefe:]=(train.iloc[:,-lefe:]-mini2)/size2
    val_train.iloc[:,-lefe:] = (val_train.iloc[:,-lefe:]-mini2)/size2


    mini3=test.iloc[:,-lefe:].min()
    size3=test.iloc[:,-lefe:].max()-test.iloc[:,-lefe:].min()
    test.iloc[:,-lefe:]=(test.iloc[:,-lefe:]-mini3)/size3

    return train, test, size, mini, val_train

#-------------------------------------------위에까지 정규화

def windo(train, test, output_window, val_train, lefe):
    test_x=tf.reshape(test.iloc[:,-lefe:].values, [1, 24*7, lefe])
    val_x = tf.reshape(val_train.iloc[:,-lefe:].values, [1, 24*7, lefe])
    val_y = tf.reshape(val_train.iloc[:,-(lefe+1)].values, [1, 24*7, 1])

    train_x=tf.reshape(train.iloc[:,-lefe:].values, [1, 24*14, lefe])
    train_y=tf.reshape(train.iloc[:,-(lefe+1)].values, [1, 24*14, 1])

    test_window_x = np.zeros((1,24,7,lefe))

    val_window_x = np.zeros((1, 24 , 7, lefe))  
    val_window_y = np.zeros((1, 24 , 7, 1))  

    train_window_x = np.zeros((1, 24 , 14, lefe))  
    train_window_y = np.zeros((1, 24 , 14, 1))  


    for j in range(0,24):
        for i in range(0,168,24):
            test_window_x[0][j][i//24] = test_x[0,j+i]

    for j in range(0,24):
        for i in range(0,168,24):
        #for i in range(0,2040,24):
            val_window_x[0][j][i//24] = val_x[0,j+i]
            val_window_y[0][j][i//24] = val_y[0,j+i]

    for j in range(0,24):
        for i in range(0,336,24):
        #for i in range(0,2040,24):
            train_window_x[0][j][i//24] = train_x[0,j+i]
            train_window_y[0][j][i//24] = train_y[0,j+i]


    new_train_x=tf.reshape(train_window_x, [-1, output_window, lefe])
    new_train_y=tf.reshape(train_window_y, [-1, output_window, 1])
    
    new_val_x=tf.reshape(val_window_x, [-1, 7, lefe])
    new_val_y=tf.reshape(val_window_y, [-1, 7, 1])

    test =tf.reshape(test_window_x, [-1, 7, lefe])
    
    return new_train_x, new_train_y, test, new_val_x, new_val_y

#--------------------------------------------------------------------------------------------------

# 08. 모델 생성

def model(xtrain, ytrain, xtest, size, mini, val_x, val_y, lefe):
    model = Sequential()
    model.add(Dense(72,input_dim=(lefe), activation="relu"))
    model.add(Dense(48,input_dim=(lefe), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(24,input_dim=(lefe), activation="relu"))
    #model.add(Dense(14,input_dim=(5), activation="relu"))
    model.add(Dense(7,input_dim=(lefe), activation="relu"))
    model.add(Dense(1))
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=100,
                                                    mode='auto')
    #컴파일 및 모델 핏

    model.compile(loss=SMAPE, optimizer="adam")
    model.fit(xtrain, ytrain, epochs=1500, batch_size=1, validation_data = (val_x, val_y), callbacks=[early_stopping])
    pred = model.predict(xtest)
    pred = pred *size + mini
    print(pred)

    return pred


output_window = 14 

#Dense모델 구동할떄------------------------------------------------------------------------------------

pred1 = []

for i in range(1,61):
    print(i)
    corr1 = corr[corr.num == i]
    di = co(corr1)
    corli = di['num{}'.format(i)]
    lefe = len(corli)
    train1, test1, val_train1 = num(train, test, i)
    train2, test2, size, mini, val_train2 = wjdrb(train1, test1, val_train1, corli, lefe)
    xtrain, ytrain, xtest, val_x, val_y= windo(train2, test2, output_window, val_train2, lefe)
    pred = model(xtrain, ytrain, xtest, size, mini, val_x, val_y, lefe)

    for i in range(0,7):
        for j in range(0,24):
            pred1.append(pred[j][i])

df = pd.DataFrame(pred1)
print(df)



df.to_csv("C:\\Users\\user\\OneDrive\\바탕 화면\\corsmape.csv", encoding="euc-kr")
# %%