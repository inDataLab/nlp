
# coding: utf-8

# Importing Model
# =============

# In[1]:

# 만들어진 모델을 불러오는데 필요한 코드
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as lgt
import re

# tokeinize - Twitter
from konlpy.tag import Twitter
twit = Twitter()

curDir = os.getcwd() # curDir = 현재 경로 / 폴더명1 / 폴더명 2 / 파일명.pkl

# model1
words1 = ['Adjective','Adverb','Conjunction','Exclamation','KoreanParticle','PreEomi','Eomi','Hashtag']
def tokenize1(doc):
    tagged = ['/'.join(t) for t in twit.pos(doc, norm=True, stem=True)]
    result = [word for word in tagged if word.split('/')[-1] in words1]
    return result
model1 = pickle.load(open(os.path.join(curDir,'model1.pkl'),'rb')) # call model
tfidf1 = pickle.load(open(os.path.join(curDir,'tfidf1.pkl'),'rb')) # call vectorizer

# model2
words2 = ['Noun','Josa','Adjective','Adverb','Exclamation','KoreanParticle','PreEomi','Eomi']
def tokenize2(doc):
    tagged = ['/'.join(t) for t in twit.pos(doc, norm=True, stem=True)]
    result = [word for word in tagged if word.split('/')[-1] in words2]
    return result
model2 = pickle.load(open(os.path.join(curDir,'model2.pkl'),'rb')) # call model
tfidf2 = pickle.load(open(os.path.join(curDir,'tfidf2.pkl'),'rb')) # call vectorizer

# model3
def tokenize3(doc):
    tagged = ['/'.join(t) for t in twit.pos(doc, norm=True, stem=True)]
    result = [word for word in tagged if word.split('/')[-1]]
    return result

model3 = pickle.load(open(os.path.join(curDir,'model3.pkl'),'rb')) # call model
tfidf3 = pickle.load(open(os.path.join(curDir,'tfidf3.pkl'),'rb')) # call vectorizer


# Read Data_Set
# =============

# In[2]:

# excel파일 열기

import pandas as pd
import os
import pickle
curDir = os.getcwd()
# path에 파일명 입력
path = curDir+r'\raw_data\youtube.xlsx'
original_file = path[path.rfind('/')+1:path.rfind('.')]
df_test=pd.read_excel(path)
h_x = df_test['contents'].as_matrix()


# In[3]:

# txt 파일 열기

import pandas as pd
import os
import pickle
curDir = os.getcwd()
# path에 파일명 입력
path = curDir+r'\raw_data\youtube_notadv.txt'
original_file = path[path.rfind('\\')+1:path.rfind('.')]
df_test=pd.read_table(path,delimiter='\t')
h_x = df_test['contents'].as_matrix()


# ### Processing...

# In[4]:

def score(model,tfidf,input_data):
    target = tfidf.transform(h_x)
    result = model.predict(target)
    result_p = model.predict_proba(target)
    df_prob = pd.DataFrame(result_p)
    result_p0 = df_prob[0]
    result_p1 = df_prob[1]
    point = round((result_p1 - result_p0),2)*100
    return point


# In[ ]:

score1 = score(model1,tfidf1,h_x) # n_gram1
score2 = score(model2,tfidf2,h_x) # n_gram2
score3 = score(model3,tfidf3,h_x) # n_gram3


# In[ ]:

result = score1/2 + score2/4 + score3/4
label = []
for i in  range(len(result)):
    if result[i] >= 50:
        label.append('POS')
    elif result[i] > -50:
        label.append('NEU')
    else:
        label.append('NEG')


# Output Data
# =============

# In[ ]:

data = {'point':result,'predict_label':label}
labeled = pd.DataFrame(data)
output = df_test.join(labeled,lsuffix='_df_test', rsuffix='labeled')


# In[32]:

output.to_excel(curDir+'/result_data/predicted_'+original_file+'.xlsx',encoding='utf-8')


# In[ ]:



