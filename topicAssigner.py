import webscraper
import lda
import gensim

"""
def assignTopics(vec, corpus):
    ltfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)
    sims = index[tfidf[vec]]
    print(list(enumerate(sims)))
"""

def topicDetector(file_name, ldamodel):
    with open('./corpus/' + file_name, 'r') as content_file:
        corpus = eval(content_file.read())
        for i in range(len(corpus)):
            topicSims = []
            for topic, sim in ldamodel.get_document_topics(corpus[i]):
                if (sim > 0.2):
                    topicSims += [(topic, sim)]
            with open('./topicSims/' + file_name, 'a') as outFile:
                outFile.write(str(topicSims))
                outFile.write('\n')

def topicsGenerator(links):
    for link in links:
        sketch = link.split('/')
        fileName = sketch[-1]
        topicDetector(fileName, gensim.models.LdaModel.load('./ldaModel/lda.model'))

#links = webscraper.getWebsites('https://www.yelp.com/search?find_desc=food&find_loc=Berkeley,+CA,+US&start=0&sortby=review_count', 4)

#topicsGenerator(links)
topicDetector("cheese-board-pizza-berkeley?osq=food", gensim.models.LdaModel.load('./ldaModel/lda.model'))
