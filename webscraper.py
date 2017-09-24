import requests
import pandas as pd
from bs4 import BeautifulSoup

def review_scraper(website):
    page = requests.get(website)
    html = BeautifulSoup(page.content, 'html.parser')
    review_count_str = html.find(class_ = \
     'review-count rating-qualifier').getText().encode('utf-8')
    review_count_num = int(review_count_str.strip().split(" ")[0])
    website.replace('?osq=food', "")
    all_reviews = []
    for i in range(1, review_count_num/20 + 2):
        review_block = html.find_all(class_ = 'review-content')
        review_text = [review.p.get_text() for review in review_block]
        all_reviews.extend(review_text)
        page = requests.get(website + "?start=" + str(i * 20))
        html = BeautifulSoup(page.content, 'html.parser')
    review_chart = pd.DataFrame({'review_text': all_reviews})
    review_chart.to_csv('./data/unfiltered/' + website + '.csv', encoding='utf-8')


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
