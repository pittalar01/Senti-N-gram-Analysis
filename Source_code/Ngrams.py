#Importing all required libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import re
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

#Reading the input file
folder = r'C:\imdb-movie-reviews-dataset\aclImdb'
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()

for f in ('test', 'train'):    
    for l in ('pos', 'neg'):
        path = os.path.join(folder, f, l)
        for file in os.listdir (path) :
            with open(os.path.join(path, file),'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]],ignore_index=True)
df.columns = ['review', 'sentiment']
#Converting to csv file
df.to_csv(r'C:\movie_reviews\movie_data.csv',index=False , encoding= 'utf-8')

#Splitting to train and test datasets
X_train,X_test,y_train,y_test= train_test_split(df['review'].values,df['sentiment'].values,train_size=0.80)

#Preprocessing: Removing html tags, Punctuation tags
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))
def text_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripp = re.sub(combined_pat, '', souped)
    try:
        clean = stripp.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripp
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    
    # tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()
 
testing = X_train[:40000]
train_result = []
for t in testing:
    train_result.append(text_cleaner(t))
#train_result
clean_df = pd.DataFrame(train_result,columns=['text'])
clean_df['target'] = y_train

testing_test = X_test[:10000]
test_result = []
for t in testing_test:
    test_result.append(text_cleaner(t))
clean_df_test = pd.DataFrame(test_result,columns=['text'])
clean_df_test['target'] = y_test

reviews_train_clean=clean_df['text']
reviews_test_clean=clean_df_test['text']
#Lemnnatization
def lemmatized_text(text):
    
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in text]

reviews_train_clean = lemmatized_text(reviews_train_clean)


#ignoring noisy data
stop_words=['in','of','at','a','the']

#unigrams
vectorizer=TfidfVectorizer(min_df=10, max_df=0.5,stop_words=stop_words,decode_error='ignore')
vectorized = vectorizer.fit_transform(reviews_train_clean)
X_test=vectorizer.transform(reviews_test_clean)
param_grid = {'C': [0.001, 0.01, 0.1,0.25, 1]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(vectorized, y_train)
lr = grid.best_estimator_
lr.fit(vectorized, clean_df['target'])
z=lr.predict(X_test)
Accuracy1= lr.score(X_test, clean_df_test['target'])
print("Unigramns Accuracy:",Accuracy1)
print("Unigramns Precision:",str(metrics.precision_score(z, clean_df_test['target'])))
print("Unigramns Recall:",metrics.recall_score(z, clean_df_test['target']))



#bigrams
vectorizer=TfidfVectorizer(min_df=10, max_df=0.5,ngram_range=(2,2),stop_words=stop_words,decode_error='ignore')
vectorized = vectorizer.fit_transform(reviews_train_clean)
X_test=vectorizer.transform(reviews_test_clean)
grid.fit(vectorized, y_train)
lr = grid.best_estimator_
lr.fit(vectorized, clean_df['target'])
z=lr.predict(X_test)
Accuracy2= lr.score(X_test, clean_df_test['target'])
print("Bigrams Accuracy:",metrics.accuracy_score(z, clean_df_test['target']))
print("Bigramns Precision:",str(metrics.precision_score(z, clean_df_test['target'])))
print("Bigrams Recall:",metrics.recall_score(z, clean_df_test['target']))



#trigrams
vectorizer=TfidfVectorizer(min_df=10, max_df=0.5,ngram_range=(3,3),stop_words=stop_words,decode_error='ignore')
vectorized = vectorizer.fit_transform(reviews_train_clean)
X_test=vectorizer.transform(reviews_test_clean)
grid.fit(vectorized, y_train)
lr = grid.best_estimator_
lr.fit(vectorized, clean_df['target'])
z=lr.predict(X_test)
Accuracy3= lr.score(X_test, clean_df_test['target'])
print("Trigrams Accuracy:",metrics.accuracy_score(z, clean_df_test['target']))
print("TrigramsPrecision:",str(metrics.precision_score(z, clean_df_test['target'])))
print("Trigrams Recall:",metrics.recall_score(z, clean_df_test['target']))



#mixture of unigrams and bigrams
vectorizer=TfidfVectorizer(min_df=10, max_df=0.5,ngram_range=(1,2),stop_words=stop_words,decode_error='ignore')
vectorized = vectorizer.fit_transform(reviews_train_clean)
X_test=vectorizer.transform(reviews_test_clean)
grid.fit(vectorized, y_train)
lr = grid.best_estimator_
lr.fit(vectorized, clean_df['target'])
z=lr.predict(X_test)
Accuracy4= lr.score(X_test, clean_df_test['target'])
print("Unigrams and Bigrams Accuracy:",metrics.accuracy_score(z, clean_df_test['target']))
print("Unigrams and Bigrams Precision:",str(metrics.precision_score(z, clean_df_test['target'])))
print("Unigrams and Bigrams Recall:",metrics.recall_score(z, clean_df_test['target']))



#mixture of unigrams, bigrams and trigrams.
vectorizer=TfidfVectorizer(min_df=10, max_df=0.5,ngram_range=(1,3),stop_words=stop_words,decode_error='ignore')
vectorized = vectorizer.fit_transform(reviews_train_clean)
X_test=vectorizer.transform(reviews_test_clean)
grid.fit(vectorized, y_train)
lr = grid.best_estimator_
lr.fit(vectorized, clean_df['target'])
z=lr.predict(X_test)
Accuracy5= lr.score(X_test, clean_df_test['target'])
print("Unigrams,Bigrams and Trigrams Accuracy:",metrics.accuracy_score(z, clean_df_test['target']))
print("Unigrams,Bigrams and Trigrams Precision:",str(metrics.precision_score(z, clean_df_test['target'])))
print("Unigrams,Bigrams and Trigrams Recall:",metrics.recall_score(z, clean_df_test['target']))



#plotting
names=['(1,1)','(2,2)','(3,3)','(1,2)','(1,3)']
values=[Accuracy1,Accuracy2,Accuracy3,Accuracy4,Accuracy5]
plt.ylim(0.8, 0.93)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()

