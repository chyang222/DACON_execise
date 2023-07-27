import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, pos_tag
import re



train = pd.read_csv("C:\\Users\\user\\Desktop\\dacon\\train.csv")
#train = train[["facts", "first_party_winner"]]
test = pd.read_csv("C:\\Users\\user\\Desktop\\dacon\\test.csv")
#test = test[["facts"]]

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

train.isna().mean().sort_values(ascending=False).plot( kind='ba', figsize=(15,5),  grid=True, color='blue', edgecolor='black', linewidth=2, rot=42)
plt.title('NaN')
plt.xlabel('Name of column')
plt.ylabel('Frac of Nan')
#plt.show() nan - 0

test.isna().mean().sort_values(ascending=False).plot( kind='bar', figsize=(15,5),  grid=True, color='blue', edgecolor='black', linewidth=2, rot=42)
plt.title('NaN')
plt.xlabel('Name of column')
plt.ylabel('Frac of Nan')
#plt.show() nan - 0

#----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 정규 표현식 함수 정의
'''
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, pos_tag
import re

def apply_regular_expression(text):
    english = re.compile('/^[a-zA-Z0-9]*$/;')  
    result = english.sub('', text)
    result = nltk.word_tokenize(result)
    tagged_tokens = nltk.pos_tag(result)  # 품사 태깅
    nouns = [word for word, pos in tagged_tokens if pos.startswith('N')]  # 명사만 추출
    nouns = [x for x in nouns if len(x) > 1]  # 한글자 키워드 제거

    return nouns

train.facts = train.facts.apply(lambda x: apply_regular_expression(x))
test.facts = test.facts.apply(lambda x: apply_regular_expression(x)) 

win = train[train.first_party_winner == 1]
fi = win[win.first_party.duplicated(keep=False)] 
se = win[win.second_party.duplicated(keep=False)] 

fi.first_party.unique()
se.second_party.unique()

win[win.second_party == "Arizona"]

lose = train[train.first_party_winner == 0]
lose[lose.first_party.duplicated(keep=False)] 
se2 = lose[lose.second_party.duplicated(keep=False)] 

se2.second_party.unique()
lose[lose.second_party == "Arizona"]

vectorizer = TfidfVectorizer()

X_train = get_vector(vectorizer, train, True)

Y_train = train["first_party_winner"]

test = get_vector(vectorizer, test, False)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)

'''
#----------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_vector(vectorizer, df, train_mode):
    if train_mode:
        X_facts = vectorizer.fit_transform(df['facts'])
    else:
        X_facts = vectorizer.transform(df['facts'])
    X_party1 = vectorizer.transform(df['first_party'])
    X_party2 = vectorizer.transform(df['second_party'])
    
    X = np.concatenate([X_party1.todense(), X_party2.todense(), X_facts.todense()], axis=1)

    return X

def extract_nouns(text): #명사만 추출
    nouns = []
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    for word, pos in tagged:
        if pos.startswith('NN'):  
            nouns.append(word)

    return ' '.join(nouns)


def count_vectorizer(vectorizer, df, train_mode):
    df.facts = df.facts.apply(extract_nouns)
    if train_mode:
        data = vectorizer.fit_transform(df.facts)
    else:
        data = vectorizer.transform(df.facts)
    
    X_party1 = vectorizer.transform(df['first_party'])
    X_party2 = vectorizer.transform(df['second_party'])

    data = np.concatenate([X_party1.todense(), X_party2.todense(), data.todense()], axis=1)

    return data


def tf_vectorizer(vectorizer, df, train_mode):

    df.facts = df.facts.apply(extract_nouns)
    if train_mode:
        data = vectorizer.fit_transform(df.facts)
    else:
        data = vectorizer.transform(df.facts)

    X_party1 = vectorizer.transform(df['first_party'])
    X_party2 = vectorizer.transform(df['second_party'])

    data = np.concatenate([X_party1.todense(), X_party2.todense(), data.todense()], axis=1)
    
    return data


def confusion(y_test, y_pred):
    confu = confusion_matrix(y_true = y_test, y_pred = y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(confu, annot=True, annot_kws={'size':15}, cmap='OrRd', fmt='.10g')
    plt.title('Confusion Matrix')
    plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

#x_tr = train[["facts"]]
x_tr = train[train.columns[:-1]]
y_tr = train[train.columns[-1]]
test1 = test

cnt_vector = CountVectorizer(min_df=2)
x_tr1 = count_vectorizer(cnt_vector,  x_tr, True)
test1 = count_vectorizer(cnt_vector, test1, False)


x_tr1 = np.asarray(x_tr1)

#####백터화 시켯는데 시킨것에서 다 0이면 없애버리자
zero_idx = []
for i in range(len(x_tr1)):
    if np.all(x_tr1[i] == 0):
        zero_idx.append(i)

#zeor_idx는 None 이제 어쩌지...... -> facts컬럼 핵심 키워드 추출로 가보자!!!  -> https://soyoung-new-challenge.tistory.com/45
#https://ebbnflow.tistory.com/292


#tf_vector = TfidfVectorizer()
#x_tr1 = tf_vectorizer(tf_vector,  x_tr, True)
#test1 = tf_vectorizer(tf_vector, test1, False)

X_train, X_test, Y_train, Y_test = train_test_split(x_tr1, y_tr, test_size=0.3, random_state=42)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_test = np.asarray(X_test)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

model = LogisticRegression().fit(X_train, Y_train)

pred = model.predict(X_test)

report = classification_report(Y_test, pred)
print(report)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#1대1조정
train['first_party_winner'].value_counts()

positive_random_idx = train[train['first_party_winner']==1].sample(829, random_state=42).index.tolist()
negative_random_idx = train[train['first_party_winner']==0].sample(829, random_state=42).index.tolist()

random_idx = positive_random_idx + negative_random_idx

x = count_vectorizer(cnt_vector, x_tr.iloc[random_idx], True)
y = y_tr[random_idx]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)

lr2 = LogisticRegression()
lr2.fit(x_train, y_train)

y_pred = lr2.predict(x_test)

report_1 = classification_report(y_test, y_pred)

print(report)
print(report_1)
confusion(y_test, y_pred)


td_test = test
td_test =  count_vectorizer(cnt_vector, td_test, False)
td_test = np.asarray(td_test)
result = lr2.predict(td_test)


submit = pd.read_csv('C:\\Users\\user\\Desktop\\dacon\\sample_submission.csv')
submit['first_party_winner'] = result
submit.to_csv('C:\\Users\\user\\Desktop\\dacon\\sample_submission.csv', index=False)


#XGboost모델링 부터가고
#일단은 모델랑 부터 가보자
#-> accuracy가 낮아도 1대1 샘플링이 점수로 더 높게 나옴 
#-> random_state의 문제 일 수 있음!

##x_train, y_train

#----------------------------------------------------------------------------------------------------------------------------------------------------------------
##validation까지 데이터 수정해서 데이터 처린


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

xg = XGBClassifier(n_estimators = 400, learning_rate = 0.1, max_depth = 3)
evals = [(x_val, y_val)]

xg.fit(x_train, y_train, early_stopping_rounds = 100, eval_metric = "logloss", eval_set = evals, verbose = True )

y_pred1 = xg.predict(x_test)

report_2 = classification_report(y_test, y_pred1)

print(report_1)

print(report_2)

confusion(y_test, y_pred1)

result = xg.predict(test)

submit = pd.read_csv('C:\\Users\\user\\Desktop\\dacon\\sample_submission.csv')
submit['first_party_winner'] = result
submit.to_csv('C:\\Users\\user\\Desktop\\dacon\\sample_submission.csv', index=False)



from pycaret.classification import *

exp_clf = setup(data = train, target = 'first_party_winner', session_id=123)
best_model = compare_models()