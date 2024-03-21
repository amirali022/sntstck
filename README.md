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
	
	obtaining historical OHLCV data of S&P 500, DJIA, NYSE and NASDAQ indices

1. Browsing, Cleaning and Analysing Textual Input [Forbes News Data](./01_1_browse_forbes_news_dataset.ipynb) and [CNBC News Data](./01_2_browse_cnbc_news_data.ipynb)

	these notebooks contains following procedures:

	- dealing with missing fields
	- creating word frequency features (TF-IDF and BoW) and visualization
	- preparing text data for sentiment analysis

2. Calculating Sentiment Score [Forbes News Data](./02_1_forbes_news_sentiment_analysis.ipynb) and [CNBC News Data](./02_2_cnbc_news_sentiment_analysis.ipynb)

	these notebooks contains following procedures:

	- calculate sentiment scores (negative, neutral, positive and compound) using VADER lexicon
	- compound scores histogram
	- plotting compound scores over time

3. Preparing Data [Forbes Sentiment Data](./03_1_data_prepration_forbes.ipynb) and [CNBC Sentiment Data](./03_2_data_prepration_cnbc.ipynb)
	
	concatenating OHLCV data with sentiment scores for S&P 500, DJIA, NYSE and NASDAQ indices

4. Predicting Without Sentiment
	
	predicting adjusted close price using historical data
	
	- S&P 500
		- [Support Vector Regression (SVR)](./04_01_01_predict_sp500_without_sentiment_SVR.ipynb)
		- [Perceptron](./04_01_02_predict_sp500_without_sentiment_Perceptron.ipynb)
		- [RNN](./04_01_03_predict_sp500_without_sentiment_RNN.ipynb)
		- [LSTM](./04_01_04_predict_sp500_without_sentiment_LSTM.ipynb)
		- [GRU](./04_01_05_predict_sp500_without_sentiment_GRU.ipynb)
		- [Bi-RNN](./04_01_06_predict_sp500_without_sentiment_BiRNN.ipynb)
		- [Bi-LSTM](./04_01_07_predict_sp500_without_sentiment_BiLSTM.ipynb)
		- [Bi-GRU](./04_01_08_predict_sp500_without_sentiment_BiGRU.ipynb)

	- DJIA [To-Do]
	- NYSE [To-Do]
	- NASDAQ [To-Do]

5. Predicting With Sentiment
	
	predicting adjusted close price using historical data and compound sentiment score

	- S&P 500
		- Forbes
			- [Support Vector Regression (SVR)](./05_01_01_01_predict_sp500_with_forbes_sentiment_SVR.ipynb)
			- [Perceptron](./05_01_01_02_predict_sp500_with_forbes_sentiment_Perceptron.ipynb)
			- [RNN]
			- [LSTM]
			- [GRU]
			- [Bi-RNN]
			- [Bi-LSTM]
			- [Bi-GRU]
		- CNBC
			- [Support Vector Regression (SVR)]
			- [Perceptron]
			- [RNN]
			- [LSTM]
			- [GRU]
			- [Bi-RNN]
			- [Bi-LSTM]
			- [Bi-GRU]

	- DJIA [To-Do]
	- NYSE [To-Do]
	- NASDAQ [To-Do]

##### Requirements

- python 3.11
- jupyter notebook

##### Libraries

- yfinance
- pandas
- sklearn
- wordcloud
- matplotlib
- nltk + vader_lexicon
- tensorflow

##### Textual data were crawled using crawler provided in [Financial Textual Data Scraper](https://github.com/amirali022/fintxt)