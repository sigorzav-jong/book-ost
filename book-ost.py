import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice

import requests
import json

from sklearn.metrics.pairwise import cosine_similarity

from selenium import webdriver
from selenium.webdriver.common.by import By
import time

from googletrans import Translator

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

st.title('Book-OST🎧')
st.divider()

data = pd.read_excel('final_data.xlsx',index_col=0)
st.header('프로젝트 기획 배경')
st.markdown('근래에 들어 **한국인의 독서량 감소**와 **젊은 층의 문해력 저하**가 사회적 문제로 떠오르고 있습니다. 책이나 신문과 같은 출판물로 정보를 습득했던 과거와 달리, 오늘날 사람들은 책 이외의 수많은 정보 매체와 미디어로부터 정보를 습득할 수 있게 되며 자연스럽게 독서량이 감소해오고 있습니다.')
st.markdown('미디어를 통한 정보 습득과 달리, 독서는 정제되지 않은 정보를 스스로 이해하고 자신의 것으로 습득하는 지적 과정을 거치기 때문에 독서가 문해력과 같은 지적 능력 발달에 매우 중요한 것으로 알려져 있습니다. 따라서 젊은 층의 문해력 저하 문제의 원인이 ‘독서량 감소’에 있다는 의견이 제기되고 있습니다.')
st.markdown('이러한 **한국인의 독서량 감소**와 **젊은 층의 문해력 저하**에 대하여, 저희 팀은 **독서에 대한 흥미를 높이고 독서를 장려할 수 있는 방안을 제시하는 것**이 두 문제의 해결 방안이 될 것이라 생각했습니다.')
st.markdown('')
st.image('멜로디책방.png')
st.markdown('')
st.markdown('''영화나 드라마처럼 **책에도 ost가 필요하다는 Jtbc 멜로디책방 프로그램**으로부터 영감을 얻어, **도서 맞춤 음악 추천 시스템**이라는 주제를 선정했습니다. 자신이 읽고 있는 책을 입력하면 책과 잘 어울리는 음악을 추천해줌으로써 **책의 감정과 내용을 음악 함께 더욱 깊이 음미하는 독서 경험을 제공**하고자 합니다. 젊은 층에게 친숙한 음악을 독서와 결합함으로써 독서에 대한 흥미와 즐거움을 더하고, 장기적으로 독서를 장려하는 하나의 문화적 서비스가 될 수 있을 것으로 기대하고 있습니다.''')

st.divider()

st.subheader('📖 책 제목을 입력해주세요. 📖')
book_title = st.text_input('예시) 날씨가 좋으면 찾아가겠어요')
if not book_title:
    st.stop()
st.success('✅ Thank You')

rest_api_key = "41d651c93152d5ec054dc828cacfa671"
url = "https://dapi.kakao.com/v3/search/book"
header = {"authorization": "KakaoAK "+rest_api_key}
querynum = {"query": book_title}

response = requests.get(url, headers=header, params = querynum)
content = response.text
책정보 = json.loads(content)['documents'][0]

book = pd.DataFrame({'title': 책정보['title'],
              'isbn': 책정보['isbn'],
              'authors': 책정보['authors'],
              'publisher': 책정보['publisher']})

target_url = 책정보['url']

text = '정보를 모으고 있는 중입니다.'
my_bar = st.progress(0, text=text)

# 옵션 생성
options = webdriver.ChromeOptions()
# 창 숨기는 옵션 추가
options.add_argument("headless")

driver = webdriver.Chrome(options=options)
driver.get(target_url)
time.sleep(5)


try :
    botton = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[2]/div[3]/a')
    botton.click()
except :
    pass
책소개 = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[2]/p')

time.sleep(3)
try :
    botton = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[5]/div[3]/a')
    botton.click()
except :
    pass
책속으로 = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[5]/p')


time.sleep(3)
try :
    botton = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[6]/div[3]/a')
    botton.click()
except :
    pass
서평 = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[6]/p')


book['책소개'] = 책소개.text
book['책속으로'] = 책속으로.text
book['서평'] = 서평.text
driver.close()

time.sleep(1)
my_bar.progress(30, text=text)


#영어 불용어 사전
stops = set(stopwords.words('english'))

def hapus_url(text):
    mention_pattern = r'@[\w]+'
    cleaned_text = re.sub(mention_pattern, '', text)
    return re.sub(r'http\S+','', cleaned_text)

#특수문자 제거
#영어 대소문자, 숫자, 공백문자(스페이스, 탭, 줄바꿈 등) 아닌 문자들 제거
def remove_special_characters(text, remove_digits=True):
    text=re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


#불용어 제거
def delete_stops(text):
    text = text.lower().split()
    text = ' '.join([word for word in text if word not in stops])
    return text
   
    
#품사 tag 매칭용 함수
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    

#품사 태깅 + 표제어 추출
def tockenize(text):
    tokens=word_tokenize(text)
    pos_tokens=nltk.pos_tag(tokens)
    
    text_t=list()
    for _ in pos_tokens:
        text_t.append([_[0], get_wordnet_pos(_[1])])
    
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word[0], word[1]) for word in text_t])
    return text

def clean(text):
    text = remove_special_characters(text, remove_digits=True)
    text = delete_stops(text)
    text = tockenize(text)
    return text

my_bar.progress(50, text=text)

translator = Translator()
for col in ['책소개', '책속으로', '서평']:
    name = col+'_trans'
    if book[col].values == '':
        book[name] = ''
        continue
    book[name] = clean(translator.translate(hapus_url(book.loc[0, col])).text)

total_text = book.loc[0, '책소개_trans'] + book.loc[0, '책속으로_trans'] + book.loc[0, '서평_trans']

df = pd.read_csv('tweet_data_agumentation.csv', index_col = 0)

tfidf_vect_emo = TfidfVectorizer()
tfidf_vect_emo.fit_transform(df["content"])

model = joblib.load('SVM.pkl')
total_text2 = tfidf_vect_emo.transform(pd.Series(total_text))
model.predict_proba(total_text2)
sentiment = pd.DataFrame(model.predict_proba(total_text2),index=['prob']).T
sentiment['감정'] = ['empty','sadness','enthusiasm','worry','love','fun','hate','happiness','boredom','relief','anger']
sentiment2 = sentiment.sort_values(by='prob',ascending=False)

my_bar.progress(60, text=text)

# audio feature랑 text 감정
audio_data = data.iloc[:,-12:-1]

sentiment_prob = sentiment['prob']
sentiment_prob.index = sentiment['감정']

audio_data.columns = ['empty', 'sadness', 'enthusiasm', 'worry', 'love', 'fun', 'hate',
       'happiness', 'boredom', 'relief', 'anger']

audio_data_1 = pd.concat([sentiment_prob,audio_data.T],axis=1).T

col = ['book']+list(data['name'])
cosine_sim_audio = cosine_similarity(audio_data_1)
cosine_sim_audio_df = pd.DataFrame(cosine_sim_audio, index = col, columns=col)

audio_sim = cosine_sim_audio_df['book']
my_bar.progress(70, text=text)

# 가사랑 text
lyrics_data = data.iloc[:,5:-12]
lyrics_data_1 = pd.concat([sentiment_prob,lyrics_data.T],axis=1).T
cosine_sim_lyrics = cosine_similarity(lyrics_data_1)
cosine_sim_lyrics_df = pd.DataFrame(cosine_sim_lyrics, index =col, columns=col)
lyrics_sim = cosine_sim_lyrics_df['book']
my_bar.progress(80, text=text)

# 키워드랑 text
keyword_data = data['key_word']
book_song_cont1 = pd.DataFrame({"text": total_text}, index = range(1))
book_song_cont2 = pd.DataFrame({"text": keyword_data})
keyword_data_1 = pd.concat([book_song_cont1, book_song_cont2], axis=0).reset_index(drop=True)

tfidf_vect_cont = TfidfVectorizer()
tfidf_matrix_cont = tfidf_vect_cont.fit_transform(keyword_data_1['text'])
tfidf_array_cont = tfidf_matrix_cont.toarray()
tfidf_df_cont = pd.DataFrame(tfidf_array_cont, columns=tfidf_vect_cont.get_feature_names_out())

cosine_sim_keyword = cosine_similarity(tfidf_array_cont)
cosine_sim_keyword_df = pd.DataFrame(cosine_sim_keyword, index = col, columns=col)
keyword_sim = cosine_sim_keyword_df['book']
my_bar.progress(90, text=text)

# 전체 유사도 계산
total_sim  = 0.8*audio_sim + 0.1*lyrics_sim + 0.1*keyword_sim

recommend_song = total_sim.sort_values(ascending=False)[1:6].inde-x
total_sim_df = pd.DataFrame(total_sim[1:])
total_sim_df = total_sim_df.reset_index()
total_sim_df.columns = ['name','book']

top_five = total_sim_df.sort_values(by='book',ascending=False)[:5]
index = total_sim_df.sort_values(by='book',ascending=False)[:5].index.sort_values()
artist = data.iloc[index][['url','name','Artist']]
top_five_df = pd.merge(artist,top_five,on='name').sort_values(by='book',ascending=False).drop_duplicates()

my_bar.progress(100, text=text)
time.sleep(1)
my_bar.empty()

st.dataframe(top_five_df)