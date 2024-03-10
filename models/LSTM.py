import os
os.environ[ "TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt

class LSTM:
	def __init__( self, neurons, input_shape):
		model = tf.keras.models.Sequential( [
			tf.keras.layers.LSTM( neurons, input_shape=input_shape, return_sequences=True),
			tf.keras.layers.LSTM( neurons, return_sequences=False),
			tf.keras.layers.Dense( 1)
		])

		model.compile(
			loss=tf.keras.losses.Huber(),
			optimizer=tf.keras.optimizers.Adam(),
			metrics=[ "mae"]
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