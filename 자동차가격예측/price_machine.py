import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from matplotlib import font_manager, rc
plt.rc('font', family='Malgun Gothic')
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import random
import os


train = pd.read_csv("C:\\Users\\user\\Desktop\\dacon2\\train.csv")
test = pd.read_csv("C:\\Users\\user\\Desktop\\dacon2\\test.csv")

#CNG, 하이브리드, 판매도시 drop
train.drop(['ID', '압축천연가스(CNG)', '하이브리드', '판매도시', '차량모델명'], axis=1, inplace=True)
test.drop(['ID', '압축천연가스(CNG)', '하이브리드', '판매도시', '차량모델명'], axis=1, inplace=True)

def categorize_mileage(mileage):
    if mileage <= 50000:
        return 0
    elif mileage <= 100000:
        return 1
    elif mileage <= 150000:
        return 2
    elif mileage <= 200000:
        return 3
    else:
        return 4

categorized_mileage = train["주행거리"].apply(categorize_mileage)
categorized_mileage2 = test["주행거리"].apply(categorize_mileage)

#train["주행거리"] = categorized_mileage
#test["주행거리"] = categorized_mileage2


label_encoder = LabelEncoder()
train["브랜드"] = label_encoder.fit_transform(train["브랜드"])
train["차량모델명"] = label_encoder.fit_transform(train["차량모델명"])
train["판매구역"] = label_encoder.fit_transform(train["판매구역"])
train["생산년도"] = label_encoder.fit_transform(train["생산년도"])
train["모델출시년도"] = label_encoder.fit_transform(train["모델출시년도"])

test["브랜드"] = label_encoder.fit_transform(test["브랜드"])
test["차량모델명"] = label_encoder.fit_transform(test["차량모델명"])
test["판매구역"] = label_encoder.fit_transform(test["판매구역"])
test["생산년도"] = label_encoder.fit_transform(test["생산년도"])
test["모델출시년도"] = label_encoder.fit_transform(test["모델출시년도"])


# 배기량 정규화
exhaust_volume = train['배기량']
road = train['주행거리']

exhaust_volume_normalized = (exhaust_volume - exhaust_volume.min()) / (exhaust_volume.max() - exhaust_volume.min())
road_normalized = (road - road.min()) / (road.max() - road.min())

train['배기량'] = exhaust_volume_normalized
train['주행거리'] = road_normalized


exhaust_volume2 = test['배기량']
road2 = test['주행거리']

exhaust_volume_normalized2 = (exhaust_volume2 - exhaust_volume2.min()) / (exhaust_volume2.max() - exhaust_volume2.min())
road_normalized2 = (road2 - road2.min()) / (road2.max() - road2.min())

test['배기량'] = exhaust_volume_normalized2
test['주행거리'] = road_normalized2


'''
columns = train.columns[:4] 
fig, axes = plt.subplots(nrows=len(columns), figsize=(10, 20))
for i, column in enumerate(columns):
    ax = axes[i]  
    ax.scatter(train[column], train["가격"], s=10, alpha=0.5)  
    ax.set_xlabel(column, fontsize=12) 
    ax.set_ylabel("Price", fontsize=12)
    ax.spines["top"].set_visible(False) 
    ax.spines["right"].set_visible(False) 
plt.tight_layout()
plt.show()


columns = train.columns[6:] 
fig, axes = plt.subplots(nrows=len(columns), figsize=(10, 20))
for i, column in enumerate(columns):
    ax = axes[i]  
    ax.scatter(train[column], train["가격"], s=10, alpha=0.5)  
    ax.set_xlabel(column, fontsize=12) 
    ax.set_ylabel("Price", fontsize=12)
    ax.spines["top"].set_visible(False) 
    ax.spines["right"].set_visible(False) 
plt.tight_layout()
plt.show()

'''

#------------------------------------------------------------------------------------------------

sns.scatterplot(data=train, x='생산년도', y='가격')
#plt.show()

corr_matrix = train.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('상관관계 행렬')
#plt.show()


produced_counts = train['생산년도'].value_counts().sort_index()
produced_counts2 = test['생산년도'].value_counts().sort_index()

#0부터 13까지 데이터 갯수가 적어서 모두 13으로 치환
train['생산년도'] = train['생산년도'].replace(range(14), 13)
test['생산년도'] = test['생산년도'].replace(range(14), 13)

#카테고리컬 형식으로 변환
#categorical_columns = ['브랜드', '판매구역', '배기량', '경유', '가솔린', '액화석유가스(LPG)', '주행거리']
#train[categorical_columns] = train[categorical_columns].astype('category')

categorical_columns = [ '경유', '가솔린', '액화석유가스(LPG)']
train[categorical_columns] = train[categorical_columns].astype('int64')
test[categorical_columns] = test[categorical_columns].astype('int64')



X = train.drop('가격', axis=1)
y = train['가격']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

exe_result = pd.DataFrame(index=X_val.index, columns=['가격'])


#------------------------------------------------------------------------------------------------
#gridsearch 하이퍼파라미터
'''
from sklearn.model_selection import GridSearchCV
xgb1 = xgb.XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X_train,
         y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
>>> {'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 6, 'min_child_weight': 4, 'n_estimators': 500, 'nthread': 4, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.7}
'''

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정


for year in X_val['생산년도'].unique():
    train_data = train[train['생산년도'] == year].drop(['생산년도'], axis=1)
    
    # X, y 데이터 분리
    X_train = train_data.drop(['가격'], axis=1)
    y_train = train_data['가격']
    
    # XGBoost 모델 학습
    model = xgb.XGBRegressor(objective='reg:squarederror',colsample_bytree=0.7, learning_rate=0.1, max_depth=7, min_child_weight=1, n_estimators=1000,  subsample=0.7)
    model.fit(X_train, y_train)
    
    # 해당 생산년도에 해당하는 test 데이터 추출
    val_data = X_val[X_val['생산년도'] == year].drop(['생산년도'], axis=1)
    
    # 예측 결과를 데이터프레임에 저장
    exe_result.loc[val_data.index, '가격'] = model.predict(val_data)

mae = mean_absolute_error(y_val, exe_result['가격'])
print("모델의 평균 절대 오차 (MAE):", mae)

#------------------------------------------------------------------------------------------------

submit = pd.read_csv('C:\\Users\\user\\Desktop\\dacon2\\sample_submission.csv')

for year in test['생산년도'].unique():
    # 해당 생산년도에 해당하는 train 데이터 추출
    train_data = train[train['생산년도'] == year].drop(['생산년도'], axis=1)
    
    # X, y 데이터 분리
    X_train = train_data.drop(['가격'], axis=1)
    y_train = train_data['가격']
    
    # XGBoost 모델 학습
    model = xgb.XGBRegressor(objective='reg:squarederror',colsample_bytree=0.7, learning_rate=0.1, max_depth=7, min_child_weight=1, n_estimators=1000,  subsample=0.7)
    model.fit(X_train, y_train)
    
    # 해당 생산년도에 해당하는 test 데이터 추출
    test_data = test[test['생산년도'] == year].drop(['생산년도'], axis=1)
    
    # 예측 결과를 submit에 추가
    submit.loc[submit.index.isin(test_data.index), '가격'] = model.predict(test_data)

print(submit)

submit.to_csv('C:\\Users\\user\\Desktop\\dacon2\\sample_submission.csv', index=False) 