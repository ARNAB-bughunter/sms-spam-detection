import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
import pickle

df = pd.read_csv('spam.csv',encoding='utf-8',encoding_errors='replace')
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.rename(columns={'v1':'target','v2':'text'},inplace=True)

encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df = df.drop_duplicates(keep='first')
df['num_characters'] = df['text'].apply(len)
df['num_words'] =  df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sentences'] =  df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    ps = PorterStemmer()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append( ps.stem(i) )
    return " ".join(y)
df['transforme_text'] = df['text'].apply(transform_text)

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transforme_text']).toarray()
Y = df['target'].values

mnb = MultinomialNB()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

mnb.fit(X_train,Y_train)
Y_pred2 = mnb.predict(X_test)
# print(accuracy_score(Y_test,Y_pred2))
# print(confusion_matrix(Y_test,Y_pred2))
# print(precision_score(Y_test,Y_pred2))
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))