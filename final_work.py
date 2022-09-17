import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))



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


vector_input = tfidf.transform([transform_text("You get free $100 on our website.")])
print("Spam") if model.predict(vector_input) == 1 else print("Not Spam")
