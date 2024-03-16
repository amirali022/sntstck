import re
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Function for creating frequecy table (dataframe) of words
def create_words_frequency( features, feature_names):
	features_df = pd.DataFrame( features)
	features_df.columns = feature_names
	sorted_features = features_df.sum( axis=0).sort_values( ascending=False)
	sorted_features = sorted_features.reset_index()
	sorted_features.columns = [ "Top Words", "Counts"]
	return sorted_features

# Function for visualizing word frequency
def generate_word_cloud( word_frequency, title):
	word_freq_dict = {}
	for word, freq in zip( word_frequency[ "Top Words"], word_frequency[ "Counts"]):
		word_freq_dict[ word] = freq

	word_cloud = WordCloud(
		width=1920,
		height=1080,
		background_color="white"
	).generate_from_frequencies( word_freq_dict)

	plt.figure()
	plt.title( title)
	plt.imshow( word_cloud)
	plt.axis( "off")
	plt.tight_layout( pad=0)

	plt.show()

# Function for creating feature vectors using TF-IDF method
def create_tfidf( df, feature_column, max_feature_size):
	tfidf_vec = TfidfVectorizer(
		preprocessor=lambda x: re.sub( r'(\d[\d\.])+', "", x.lower()),
		sublinear_tf=True,
		min_df=2,
		norm="l2",
		encoding="latin-1",
		ngram_range=( 1, 3),
		stop_words=list( text.ENGLISH_STOP_WORDS),
		max_features=max_feature_size
	)

	features = tfidf_vec.fit_transform( df[ feature_column]).toarray()

	return features, tfidf_vec

# Function for creating feature vectors using BoW method
def create_bow( df, feature_column, max_feature_size):
    counter_vec = CountVectorizer(
		preprocessor=lambda x: re.sub( r'(\d[\d\.])+', "", x.lower()),
        encoding="latin-1",
        ngram_range=( 1, 3),
        stop_words=list( text.ENGLISH_STOP_WORDS),
        max_features=max_feature_size
    )

    features = counter_vec.fit_transform( df[ feature_column]).toarray()

    return features, counter_vec