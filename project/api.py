from flask import Flask, request,jsonify
from flask_restful import Resource, Api
import browserhistory as bh
import pickle


from sklearn.feature_extraction.text import CountVectorizer

import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

app = Flask(__name__)
api = Api(app)

class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=False ):
        review_text = BeautifulSoup(review).get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(words)

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( KaggleWord2VecUtility.review_to_wordlist( raw_sentence, \
                  remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences

app = Flask(__name__) #name of the module flask will know wherwe to look for static modeul

# load trained classifier
clf_path = 'G:/flask_p/sentiment_analysis'
with open(clf_path, 'rb') as f:
    model = pickle.load(f)
dict_obj = bh.get_browserhistory()

test_var = dict_obj['chrome'][123][1]

y_clean_test = []

y_clean_test = []

g_clean_test = []

YouTube = []
google_search = []

clean_test = []

for i in range(len(dict_obj['chrome'])):
    if "YouTube" in dict_obj['chrome'][i][1]:
        YouTube.append(i)
    else:
        google_search.append(i)

test_var = google_search[1]


# for i in range(0,len(YouTube)):
#     y_clean_test.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(YouTube[i],True)))
# for i in range(0,len(google_search)):
#     g_clean_test.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(google_search[i],True)))

for i in range(0,len(dict_obj['chrome'])):
    clean_test.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(dict_obj['chrome'][i][1],True)))

vectorizer = CountVectorizer(analyzer="word",\
                            tokenizer = None, \
                            stop_words = None, \
                            max_features = 5000)
test_data_features = vectorizer.fit_transform(clean_test)

test_data_features = test_data_features.toarray()

# test_var = len(test_data_features[0])






# vectorizer = CountVectorizer(analyzer="word",\
#                             tokenizer = None, \
#                             stop_words = None, \
#                             max_features = 5000)
# y_test_data_features = vectorizer.fit_transform(y_clean_test)

# # g_test_data_features = vectorizer.fit_transform(g_clean_test)

# y_test_data_features = y_test_data_features.toarray()

# test_var2 = len(y_test_data_features[0])

# vectorizer = CountVectorizer(analyzer="word",\
#                             tokenizer = None, \
#                             stop_words = None, \
#                             max_features = 5000)

# g_test_data_features = vectorizer.fit_transform(g_clean_test)

# g_test_data_features = g_test_data_features.toarray()

# test_var3 = len(g_test_data_features[0])


# y_test_data_features = y_test_data_features.toarray()


# g_test_data_features = g_test_data_features.toarray()
# temp = int(test_data_features[0][0])

# y_results = model.predict(y_test_data_features)

# g_results = model.predict(g_test_data_features)

# y_results = y_results.tolist()

# g_results = g_results.tolist()

results = model.predict(test_data_features)

results = results.tolist()

y_results =[]
g_results =[]


for i in range(len(results)):
    if i in YouTube:
        y_results.append(results[i])
    else:
        g_results.append(results[i])


y_results = int((y_results.count(1)/len(y_results)) * 100)

g_results = int(((g_results.count(1)-100)/len(g_results)) * 100)

# test1 = vectorizer.get_feature_names()
# test1 = test1[456]
# temp = int(results[0])




class apitest(Resource):
    def get(self):
        return {'test':[y_results]}


api.add_resource(apitest,'/api')


if __name__ == '__main__':
    app.run(debug=True)