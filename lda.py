import webscraper
import nltk
from string import maketrans
import string
from nltk.corpus import stopwords
import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re
from time import time
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models, similarities


links = webscraper.getWebsites('https://www.yelp.com/search?find_desc=food&find_loc=Berkeley,+CA,+US&start=0&sortby=review_count', 4)

stop_words = [word.encode('utf-8') for word in stopwords.words('english')]

stop_words.extend(["pizza", "good", "crust", "one", "berkeley", "good", "place", "get", \
"also", "pretty", "would", "really", "food", "ive", "well", "loved", "dish", "\xa0the", \
"\xa0i", "experience", "still"])
"""stop_words.extend(["pizza", "good", "crust", "one", "berkeley", "good", "place", "get", \
"also", "pretty", "would", "really", "food", "ive", "like", "great", "\xa0the", "little", "also",\
"best", "restaurant", "definitely", "always", "get", "us", "got", "dont",\
"two", "top", "want", "hot", "toss", "try", "came", "im", "love", "super", "youre", "bay", \
"delicious", "favorite", "come", "nice", "didnt", "much", "find", "even", "cheese", \
"give", "amazing", "next", "probably", "amazing", "enjoy", "slice", "slices", "sauce", "day", \
"go"])"""

def createCorpusForWebsite(link):
    file_list = []
    sketch = link.split('/')
    fileName = './data/filtered/' + 'filtered' + sketch[-1] + '.csv'
    with open(fileName, 'r') as content_file:
        reviews = content_file.read().split('\n')
        file_list.extend([review.split(' ') for review in reviews]);
    dictionary = corpora.Dictionary(file_list)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in file_list]
    with open('./corpus/' + sketch[-1], 'w') as outFile:
        outFile.write(str(doc_term_matrix))

#result, corpus = lda(links, "All", 7)
def createCorpusForGroup(links):
    for link in links:
        createCorpusForWebsite(link)

def ridExtraWords(links):
    for link in links:
        sketch = link.split('/')
        inputFile = sketch[-1] + '.csv'
        outputFile = 'filtered' + sketch[-1] + '.csv'
        with open('./data/unfiltered/' + inputFile, 'r') as inFile, open('./data/filtered/' + outputFile, 'w') as outFile:
            for line in inFile.readlines():
                index = line.find(',')
                line = line[index + 1:]
                ridStopWords = [word for word in line.lower().translate(None, string.punctuation).split() if word not in stop_words]
                fixedLines = " ".join(ridStopWords)
                outFile.write(fixedLines + '\n')

def docMatrix(links):
    file_list = []
    for link in links:
        sketch = link.split('/')
        fileName = './data/filtered/' + 'filtered' + sketch[-1] + '.csv'
        with open(fileName, 'r') as content_file:
            reviews = content_file.read().split('\n')
            file_list.extend([review.split(' ') for review in reviews]);
    dictionary = corpora.Dictionary(file_list)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in file_list]
    return dictionary, doc_term_matrix

def indivReviews(links):
    file_list = []
    for link in links:
        sketch = link.split('/')
        fileName = './data/filtered/' + 'filtered' + sketch[-1] + '.csv'
        with open(fileName, 'r') as content_file:
            reviews = content_file.read().split('\n')
            file_list.extend(reviews);
    return file_list

def lda(links, rest, num):
    ridExtraWords(links)
    dictionary, matrix = docMatrix(links)
    lda_obj = gensim.models.ldamodel.LdaModel
    ldamodel = lda_obj(matrix, num_topics=num, id2word = dictionary, passes=10)
    result = ldamodel
    result.save('lda.model')
    with open("results.csv", 'a') as result_file:
        result_file.write("Stopwords: " + " ".join(stop_words) + "\n" + "NumTopics: " + str(num) + "\n")
        result_file.write("Restaurant: " + str(rest) + "\n")
        result_file.write(str(ldamodel.print_topics(num_topics = num, num_words = 7)) + "\n" + "\n")
    #return ldamodel
    return ldamodel, matrix

#result = lda(links, "All", 7)
