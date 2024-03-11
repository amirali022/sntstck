import os
os.environ[ "TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Bidirectional, Dense
from keras.models import Sequential
from keras.losses import Huber
from keras.optimizers import Adam
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt

class BiLSTM:
	def __init__( self, neurons, input_shape):
		model = Sequential( [
			Input( shape=input_shape),
			Bidirectional( LSTM( neurons, return_sequences=True)),
			Bidirectional( LSTM( neurons, return_sequences=False)),
			Dense( 1)
		])

		model.compile(
			loss=Huber(),
			optimizer=Adam()
		)

		self.model = model
		self.history = None

	def summary( self):
		self.model.summary()

	def fit( self, input, label, epochs, batch_size):
		self.history = self.model.fit(
			input,
			label,
			epochs=epochs,
			batch_size=batch_size
		)

	def plot_loss( self):
		if( self.history is not None):
			plt.figure()
			plt.plot( self.history.history[ "loss"])
			plt.xlabel( "Epochs")
			plt.title( "Training Loss")
			plt.show()
		else:
			print( "You Need to Call fit method first")

	def predict( self, input):
		y_pred = self.model.predict( input)

		return y_pred