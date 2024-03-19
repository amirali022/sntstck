from keras.layers import Dense
from keras.models import Sequential
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class Perceptron:
	def __init__( self, input_shape):
		model = Sequential( [
			Dense( 1, input_shape=input_shape)
		])

		model.compile(
			loss=MeanSquaredError(),
			optimizer=Adam()
		)

		self.model = model
		self.history = None

	def summary( self):
		self.model.summary()

	def fit( self, input, target, epochs, batch_size, verbose=0):
		self.history = self.model.fit(
			input,
			target, 
			epochs=epochs,
			batch_size=batch_size,
			verbose=verbose
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
		y_pred = self.model.predict( input, verbose=0)

		return y_pred