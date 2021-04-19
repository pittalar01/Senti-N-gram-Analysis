# Senti-N-gram-Analysis
Sentiment analysis is the automated process of analyzing text data and classifying opinions as negative or positive. In this project we aim to use Senti-N gram Analysis on a set of movie reviews given by reviewers and try to understand what their overall reaction to the movie.

**Dataset:** The dataset used for this task was collected from Large Movie Review Dataset which was used by the AI department of Stanford University for the associated publication. The dataset contains 50,000 examples collected from IMDb where each review is labelled with the rating of the movie. 

**Methods Used:** 
1. Scikit-learn: Used scikit for implementing Logistic Regression algorithm
2. Data Preprocessing:One necessary pre-processing step is removal of HTML tags, punctuation marks. This is done using Python libraries
3. Stemming: We performed stemming, using SnowballStemmer only when working with unigrams because in case of bigrams and trigrams, stemming actually changed the meaning of few words. 
4. Lemmatization: After the pre-processing step we perform lemmatization using WordNetLemmatizer.
5. TfidfVectorizer: Machine learning algorithms cannot work with raw text directly. Rather, the text must be converted into vectors of numbers. We make use of Scikit library that provides several transformers for tf-idf transformation such as count vectorizer, tfidf transformer and tfidf vectorizer. There are two approaches for transforming text data to tf-idf data.
6. Logistic Regression: Logistic Regression is one of the most simple and commonly used Machine Learning algorithms for two-class classification. Implemented GridSearchCV for hyper-parameter tuning.

We performed Logistic Regression on different n-grams and on different combination of n-grams. 
