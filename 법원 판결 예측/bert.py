import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train = pd.read_csv("C:\\Users\\user\\Desktop\\dacon\\train.csv")
test = pd.read_csv("C:\\Users\\user\\Desktop\\dacon\\test.csv")

  
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

def con_cat(df):
    cat_result = pd.DataFrame()
    for i in df.columns:
        cat_result = pd.concat([cat_result, df[i].apply(extract_nouns)], axis=1)
    
    return cat_result

def replace_names(row):
    facts = row['facts']
    first_party = row['first_party']
    second_party = row['second_party']

    facts = facts.replace(first_party, "first_team")
    facts = facts.replace(second_party, "second_team")
    
    #first_party = first_party.replace(first_party, "3")
    #second_party = second_party.replace(second_party, "4")

    return pd.Series([first_party, second_party, facts], index=['first_party', 'second_party', 'facts'])

x_tr = con_cat(train[train.columns[1:-1]])
y_tr = train[[train.columns[-1]]]
x_tr = x_tr.apply(replace_names, axis=1)

test1 = con_cat(test[test.columns[1:]])
test1 = test1.apply(replace_names, axis=1)

sentences = ["[CLS] " + str(s) + " [SEP]" for s in x_tr.facts]

labels = y_tr.values

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(s) for s in sentences]

print(sentences[0])  #토크나이징 전
print(tokenized_texts[0]) #토크나이징 후

MAX_LEN = 128 #최대 시퀀스 길이 설정
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

attention_masks = []

for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)


import torch

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                    labels, 
                                                                                    random_state=2000, 
                                                                                    test_size=0.1)
                                                
train_masks, validation_masks, _, _ = train_test_split(attention_masks, 
                                                       input_ids,
                                                       random_state=2000, 
                                                       test_size=0.1)     
                                                       
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)        


from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 100

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


n_devices = torch.cuda.device_count()
print(n_devices)

for i in range(n_devices):
    print(torch.cuda.get_device_name(i))


if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.cuda()
