## Predicting Stock Prices Using Historical Price Data, Sentiment Analysis and Deep Learning

#### *Historical Data:* Open, High, Low, Close, Adjusted Close Price and Volume of:
- [S&P 500](./data/input/sp500.csv)
- [DJIA](./data/input/djia.csv)
- [NASDAQ](./data/input/nasdaq.csv)
- [NYSE](./data/input/nyse.csv)

#### *Textual Data:* News Headlines of:
- [Forbes.com](./data/input/forbes-news.csv) (title and description)
- [CNBC.com](./data/input/cnbc-news.zip) (title, description, tags, keyPoints)

> Historical and textual data ranges from 2018-01-01 to 2024-03-14

#### Prediction Models

- Perceptron (Single-Neuron Neural Network with Linear Activation)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM

#### Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

#### Steps for Predicting

0. [Preparing Historical Data](./00_download_historical_data.ipynb)
	- obtaining historical OHLCV data of S&P 500, DJIA, NYSE and NASDAQ indices

1. Browsing, Cleaning and Analysing Textual Input [Forbes News Data](./01_1_browse_forbes_news_dataset.ipynb) and [CNBC News Data](./01_2_browse_cnbc_news_data.ipynb)
	- dealing with missing fields
	- creating word frequency features and visualization
	- preparing text data for sentiment analysis

2. [Calculating Sentiment Score](./02_sentiment_analysis.ipynb)
	- calculate sentiment scores (neg, neu, pos, compound) using VADER lexicon
	- plotting compound scores histogram

3. [Preparing Data](./03_data_prepration.ipynb)
	- concatenating OHLCV data with sentiment scores for S&P 500, DJIA, NYSE and NASDAQ indices

4. [Predicting Without Sentiment](./04_predict_sp500_without_sentiment.ipynb)
	- predicting S&P 500 Index using Perceptron, LSTM and Bi-LSTM model without including sentiment scores

5. [Predicting With Sentiment](./05_predict_sp500_with_sentiment.ipynb)
	- predicting S&P 500 Index using Perceptron, LSTM and Bi-LSTM Model with compound sentiment score

##### Requirements

- python 3.11

##### Libraries

- yfinance
- pandas
- sklearn
- wordcloud
- matplotlib
- nltk + vader_lexicon
- tensorflow

##### Textual data were crawled using crawler provided in [Financial Textual Data Scraper](https://github.com/amirali022/fintxt)