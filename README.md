## Predicting Stock Prices Using Historical Price Data, Sentiment Analysis and Deep Learning

#### Historical Data: S&P 500, DJIA, NASDAQ, NYSE

#### Textual Data: News Headlines from Forbes.com

#### Prediction Models:

- Long Short-Term Memory (LSTM)
- Perceptron (Single-Neuron Neural Network with Linear Activation)

#### Evaluation Metrics:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

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

3. [Preparing Data](./03_data_prepration.ipynb)
	- concatenating OHLCV data with sentiment scores for S&P 500, DJIA, NYSE and NASDAQ indices

4. [Predicting Without Sentiment](./04_predict_sp500_without_sentiment.ipynb)
	- predicting S&P 500 Index using LSTM and Perceptron model without including sentiment scores

5. [Predicting With Sentiment](./05_predict_sp500_with_sentiment.ipynb)
	- predicting S&P 500 Index using LSTM and Perceptron Model with compound sentiment score

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