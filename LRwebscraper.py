import requests
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from string import maketrans
import string
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re
from time import time
from nltk.stem.wordnet import WordNetLemmatizer

# the highest rating to consider reviews for
# DO NOT MAKE MAX_ITER HIGH WHEN RATING_THRESHOLD IS 5
RATING_THRESHOLD = 3
METHOD_NAME = "lda"
NUM_TOPICS = 3 # LDA
NUM_COMPONENTS = 2 # PCA
LEARNING_DECAY = 0.7 # should be in the interval (0.5, 1.0]
MAX_ITER = 20
NUM_TOP_WORDS = 10
NGRAM_RANGE = (2, 3)
def indivReviews(links):
    file_list = []
    for link in links:
        sketch = link.split('/')
        fileName = './data/lowRatingFiltered/' + 'filtered' + sketch[-1] + '.csv'
        with open(fileName, 'r') as content_file:
            reviews = content_file.read().split('\n')
            file_list.extend(reviews);
    return file_list

def review_scraper(website):
    page = requests.get(website)
    html = BeautifulSoup(page.content, 'html.parser')
    review_count_str = html.find(class_ = \
     'review-count rating-qualifier').getText().encode('utf-8')
    review_count_num = int(review_count_str.strip().split(" ")[0])
    name = website.split('/')[-1]
    website = website.replace('?osq=food', "")
    all_reviews = []
    print(review_count_num)
    for i in range(1, review_count_num/20 + 2):
        review_content = html.find_all(class_ = 'review-content')
        for review in review_content:
            if (review.find(class_ = 'i-stars i-stars--regular-2 rating-large') \
            or review.find(class_ = 'i-stars i-stars--regular-1 rating-large')):
                all_reviews += [review.p.get_text()]
        page = requests.get(website + "?start=" + str(i * 20))
        html = BeautifulSoup(page.content, 'html.parser')
    review_chart = pd.DataFrame({'review_text': all_reviews})
    review_chart.to_csv('./data/lowRating/' + name + '.csv', encoding='utf-8')




stop_words = [word.encode('utf-8') for word in stopwords.words('english')]

stop_words.extend(["pizza", "good", "crust", "one", "berkeley", "good", "place", "get", \
"also", "pretty", "would", "really", "food", "ive", "like", "great", "\xa0the", "little", "also",\
"best", "restaurant", "definitely", "always", "get", "us", "got", "dont",\
"two", "top", "want", "hot", "toss", "try", "came", "im", "love", "super", "youre", "bay", \
"delicious", "favorite", "come", "nice", "didnt", "much", "find", "even", "cheese", \
"give", "amazing", "next", "probably", "amazing", "enjoy", "slice", "slices", "sauce", \
"day", "go", "board", "\xa0i", "know", "nazzi", "live", "music", "call", "ok", "trust" \
"sucks", "gave", "sienfeild", "rather", "cheeseboard", "sucks", "trying", "hard"])
def getWebsites(link, page_num):
    links = []
    for i in range(1,page_num + 1):
        page = requests.get(link)
        html = BeautifulSoup(page.content, 'html.parser')
        website_link_class = html.find_all(class_ = 'biz-name js-analytics-click')
        website_link_href = [website.get('href').encode('utf-8') for website in website_link_class][1:]
        website_link = ['https://www.yelp.com' + website for website in website_link_href]
        links.extend(website_link)
        link = link.replace('start=' + str((i - 1) * 10), 'start=' + str(i * 10))
    return links;


def ridExtraWords(links):
    for link in links:
        sketch = link.split('/')
        inputFile = sketch[-1] + '.csv'
        outputFile = 'filtered' + sketch[-1] + '.csv'
        with open('./data/lowRating/' + inputFile, 'r') as inFile, open('./data/lowRatingFiltered/' + outputFile, 'w') as outFile:
            for line in inFile.readlines():
                index = line.find(',')
                line = line[index + 1:]
                ridStopWords = [word for word in line.lower().translate(None, string.punctuation).split() if word not in stop_words]
                normalized = normalized = " ".join(lemma.lemmatize(word) for word in ridStopWords)
                #fixedLines = " ".join(ridStopWords)
                #outFile.write(fixedLines + '\n')
                outFile.write(normalized + '\n')


def docMatrix(links):
    file_list = []
    for link in links:
        sketch = link.split('/')
        fileName = './data/lowRatingFiltered/' + 'filtered' + sketch[-1] + '.csv'
        with open(fileName, 'r') as content_file:
            reviews = content_file.read().split('\n')
            file_list.extend([review.split(' ') for review in reviews]);
    dictionary = corpora.Dictionary(file_list)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in file_list]
    return dictionary, doc_term_matrix

def lda(links, rest, num):
    ridExtraWords(links)
    dictionary, matrix = docMatrix(links)
    lda_obj = gensim.models.ldamodel.LdaModel
    ldamodel = lda_obj(matrix, num_topics=num, id2word = dictionary, passes=10)
    result = ldamodel
    with open("lowRatingResults.csv", 'a') as result_file:
        result_file.write("Stopwords: " + " ".join(stop_words) + "\n" + "NumTopics: " + str(num) + "\n")
        result_file.write("Restaurant: " + str(rest) + "\n")
        result_file.write(str(ldamodel.print_topics(num_topics = num, num_words = 7)) + "\n" + "\n")
    return ldamodel

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % (topic_idx + 1))
        print(" ".join([feature_names[i] # + "," + str(topic[i])
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    return [(feature_names[i], topic[i])
                        for i in topic.argsort()[:-n_top_words - 1:-1]]
vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=NGRAM_RANGE)
transcripts = indivReviews(["https://www.yelp.com/biz/tacubaya-berkeley?osq=food"])
X = vectorizer.fit_transform(transcripts)
X = X.toarray()
lda = LDA(n_topics=NUM_TOPICS,
          doc_topic_prior=50 / NUM_TOPICS,
          topic_word_prior=0.1,
          learning_decay=LEARNING_DECAY,
          max_iter=MAX_ITER
)
t0 = time()
lda.fit_transform(X)
print("lda fit done in %0.3fs" % (time() - t0))
word_weight_list = print_top_words(lda, vectorizer.get_feature_names(), NUM_TOP_WORDS)

#result = lda(["https://www.yelp.com/biz/tacubaya-berkeley?osq=food"], "Tacubya", 2)
