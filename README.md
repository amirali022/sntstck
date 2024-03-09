## Predicting Stock Prices Using Historical Price Data, Sentiment Analysis and Deep Learning

#### Historical Data: S&P 500, DJIA, NASDAQ, NYSE

#### Textual Data: News Headlines from Forbes.com

#### Deep Learning Model: Long Short-Term Memory (LSTM)

#### Steps for Predicting

0. [Preparing Historical Data](./00_download_historical_data.ipynb)
	- obtaining historical OHLCV data of S&P 500, DJIA, NYSE and NASDAQ indices

1. [Browsing, Cleaning and Analysing Textual Input](./01_browse_dataset.ipynb)
	- dealing with missing fields
	- creating word frequency features and visualization
	- preparing text data for sentiment analysis

2. [Calculating Sentiment Score](./02_sentiment_analysis.ipynb)
	- calculate sentiment scores (neg, neu, pos, compound) using VADER lexicon
	- plotting compound scores histogram

##### Requirements

- python 3.11

##### Libraries

- yfinance
- pandas
- sklearn
- wordcloud
- matplotlib

##### Textual data were crawled using crawler provided in [Financial Textual Data Scraper](https://github.com/amirali022/fintxt)