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
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, pos_tag
import re



train = pd.read_csv("C:\\Users\\user\\Desktop\\dacon\\train.csv")
#train = train[["facts", "first_party_winner"]]
test = pd.read_csv("C:\\Users\\user\\Desktop\\dacon\\test.csv")
#test = test[["facts"]]

train['facts'] = train['first_party']  + ', ' + train['second_party'] + ', ' + train['facts']
test['facts'] = test['first_party']  + ', ' +  test['second_party'] + ', ' + test['facts']


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
    stop_words_list = stopwords.words('english')
    nouns = []
    result = []
    text = re.sub('[^a-zA-Z0-9]',' ',text).strip() 
    tokens = nltk.word_tokenize(text)
    
    for word in tokens:
        if word not in stop_words_list:
            result.append(word)

    tagged = nltk.pos_tag(result)
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


#train['doc_len'] = train.facts.apply(lambda words: len(words.split()))

def plot_doc_lengths(dataframe):
    mean_seq_len = np.round(dataframe.doc_len.mean()).astype(int)
    sns.distplot(tuple(dataframe.doc_len), hist=True, kde=True, label='Document lengths')
    plt.axvline(x=mean_seq_len, color='k', linestyle='--', label=f'Sequence length mean:{mean_seq_len}')
    plt.title('Document lengths')
    plt.legend()
    plt.show()
    print(f" 가장 긴 문장은 {train['doc_len'].max()} 개의 단어를, 가장 짧은 문장은 {train['doc_len'].min()} 개의 단어를 가지고 있습니다.")

#plot_doc_lengths(train)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
text_len = train[train['first_party_winner']==1]['facts'].map(lambda x: len(x))
ax1.hist(text_len, color='red')
ax1.set_title('Positive Reviews')
ax1.set_xlabel('length of samples')
ax1.set_ylabel('number of samples')
print('first_party_winner 1의 평균 길이 :', np.mean(text_len))

text_len = train[train['first_party_winner']==0]['facts'].map(lambda x: len(x))
ax2.hist(text_len, color='blue')
ax2.set_title('Negative Reviews')
fig.suptitle('Words in texts')
ax2.set_xlabel('length of samples')
ax2.set_ylabel('number of samples')
print('first_party_winner 0의 평균 길이 :', np.mean(text_len))
plt.show()


#----------------------------------------------------------------------------------------------------------------------------------------------------------------
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

#x_tr = train[["facts"]]
x_tr = train[train.columns[-2]]
y_tr = train[[train.columns[-1]]]
test1 = test[test.columns[-1]]

#positive_random_idx = train[train['first_party_winner']==1].sample(829, random_state=42).index.tolist()
#negative_random_idx = train[train['first_party_winner']==0].sample(829, random_state=42).index.tolist()

#random_idx = positive_random_idx + negative_random_idx

#x_tr = x_tr.iloc[random_idx]
#y_tr= y_tr.iloc[random_idx]


x_tr = x_tr.apply(extract_nouns)
test1 = test1.apply(extract_nouns)


tokenizer = Tokenizer() # 토큰화
tokenizer.fit_on_texts(x_tr) # 단어 인덱스 구축


threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수 = 10328개
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.

for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value 

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1 # 카운트
        rare_freq = rare_freq + value


print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)


vocab_size = total_cnt

tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(x_tr)
X_train = tokenizer.texts_to_sequences(x_tr)
X_train[:3]

X_test = tokenizer.texts_to_sequences(test1)
X_test[:3]

print(len(X_train))
print(len(X_test))


def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

max_len = 175
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)
y_train = y_tr.values
y_train = np.asarray(y_train).astype('float32')

'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

'''
word_to_index = tokenizer.word_index

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)

model = LogisticRegression().fit(X_train, y_train)

pred = model.predict(X_val)

report = classification_report(y_val, pred)
print(report)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers


embedding_dim = 100

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(16))
model.add(Dense(8))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_acc:', mode='max', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
model.compile(optimizer= optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, callbacks=[es, mc], batch_size=24, validation_data= (X_val, y_val))


from sklearn.metrics import f1_score
score, accuracy = model.evaluate(X_val, y_val, batch_size=24)
print('Val score:', score)
print('Val accuracy:', accuracy)

# F1 score
predictions = model.predict(X_val)
predictions = np.argmax(predictions, axis=1)
f1score = f1_score(y_val, predictions, average='macro')
print('val f1 : ', f1score)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------

def vis(history,name) :
    plt.title(f"{name.upper()}")
    plt.xlabel('epochs')
    plt.ylabel(f"{name.lower()}")
    value = history.history.get(name)
    val_value = history.history.get(f"val_{name}",None)
    epochs = range(1, len(value)+1)
    plt.plot(epochs, value, 'b-', label=f'training {name}')
    if val_value is not None :
        plt.plot(epochs, val_value, 'r:', label=f'validating {name}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.05, 1.2) , fontsize=10 , ncol=1)

def plot_history(history) :
    key_value = list(set([i.split("val_")[-1] for i in list(history.history.keys())]))
    plt.figure(figsize=(12, 4))
    for idx , key in enumerate(key_value) :
        plt.subplot(1, len(key_value), idx+1)
        vis(history, key)
    plt.tight_layout()
    plt.show()


plot_history(history)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

load_to_model = load_model('best_model.h5')
y_pred = load_to_model.predict(X_test)


ze = 0
fi = 0
for i in range(len(y_pred)):
  if y_pred[i] < 0.6:
    y_pred[i] = 0
    ze += 1
  else:
    y_pred[i]=1
    fi += 1


submit = pd.read_csv('C:\\Users\\user\\Desktop\\dacon\\sample_submission.csv')
submit['first_party_winner'] = y_pred
submit.to_csv('C:\\Users\\user\\Desktop\\dacon\\sample_submission.csv', index=False)


'''

model = Sequential()
model.add(layers.Dense(16, activation = "relu", input_shape=(100, )))
model.add(layers.Dense(16, activation = "relu"))
model.add(Dense(1, activation='sigmoid'))

embedding_dim = 100

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Dense(128))
model.add(LSTM(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_acc:', mode='max', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
model.compile(optimizer= optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

'''