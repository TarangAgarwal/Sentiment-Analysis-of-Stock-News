from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'GOOG', 'FB']

news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    
    html = BeautifulSoup(response, 'html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

parsed_data = []

for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text
        timestamp = row.td.text.split(' ')

        if len(timestamp) == 1:
            time = timestamp[0]
        else:
            date = timestamp[0]
            time = timestamp[1]

        parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns=['TICKER', 'DATE', 'TIME', 'TITLE'])

vader = SentimentIntensityAnalyzer()

find_polarity = lambda title: vader.polarity_scores(title)['compound']
df['COMPOUND'] = df['TITLE'].apply(find_polarity)
df['DATE'] = pd.to_datetime(df.DATE).dt.date

plt.figure(figsize=(10,8))

mean_df = df.groupby(['TICKER', 'DATE']).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs('COMPOUND', axis='columns').transpose()

mean_df.plot(kind="bar")
plt.show()