# Senti-N-gram-Analysis
Sentiment analysis is the automated process of analyzing text data and classifying opinions as negative or positive. This project uses Senti-N gram Analysis on a set of movie reviews given by reviewers and try to understand what their overall reaction to the movie.

**Dataset:** The dataset used for this task was collected from Large Movie Review Dataset which was used by the AI department of Stanford University for the associated publication. The dataset contains 50,000 examples collected from IMDb where each review is labelled with the rating of the movie. 

**Libraries Used:** scikit-learn, numpy, pandas, matplotlib, nltk, bs4

**Methods Used:** 

•	Data Preprocessing: Removed HTML tags, punctuation marks as part of data preprocessing using Python libraries.

•	Stemming: Performed stemming using SnowballStemmer only for the unigrams models. It is not implemented on bigrams and trigrams model as stemming actually changed the meaning of few words.

•	Lemmatization: After the pre-processing step, performed lemmatization using WordNetLemmatizer.

•	TfidfVectorizer: Machine learning algorithms cannot work with raw text directly. Rather, the text must be converted into vectors of numbers. So, used scikit library that provides several transformers for tf-idf transformation such as count vectorizer, tfidf transformer and tfidf vectorizer. 

•	Logistic Regression: Trained the models using Logistic Regression on different n-grams and on different combination of n-grams. Implemented GridSearchCV for hyper-parameter tuning.
