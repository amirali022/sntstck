from keras.layers import GRU, Dense
from keras.models import Sequential
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class GatedRecurrentUnit:
	def __init__( self, neurons, batch_size, window_size, input_dim, stateful=True, unroll=True):
		self.neurons = neurons
		self.batch_size = batch_size
		self.window_size = window_size
		self.input_dim = input_dim
		self.stateful = stateful
		self.unroll = unroll
		
		b = self.batch_size if self.stateful else None
		
		batch_input_shape = ( b, self.window_size, self.input_dim)

		model = Sequential( [
			GRU(
				neurons,
				batch_input_shape=batch_input_shape,
				stateful=self.stateful,
				return_sequences=True,
				unroll=self.unroll
			),
			GRU(
				neurons,
				stateful=self.stateful,
				return_sequences=False,
				unroll=self.unroll
			),
			Dense( 1)
		])

		model.compile(
			loss=MeanSquaredError(),
			optimizer=Adam()
		)

		self.model = model
		self.history = None

	def summary( self):
		self.model.summary()

	def fit( self, input, target, epochs, verbose=0):
		self.history = self.model.fit(
			input,
			target,
			epochs=epochs,
			batch_size=self.batch_size,
			shuffle=not self.stateful,
			verbose=verbose
		)

		if self.stateful:
			self.model.reset_states()

	def plot_loss( self):
		if( self.history is not None):
			plt.figure()
			plt.plot( self.history.history[ "loss"])
			plt.xlabel( "Epochs")
			plt.title( "Training Loss")
			plt.show()
		else:
			print( "You Need to Call fit method first")

	def predict( self, input, batch_size=1, verbose=0):
		model = self.model

		if self.stateful:
			lstm = GatedRecurrentUnit(
				neurons=self.neurons,
				batch_size=batch_size,
				window_size=self.window_size,
				input_dim=self.input_dim,
				stateful=self.stateful,
				unroll=self.unroll
			)

			lstm.model.set_weights( self.model.get_weights())

			model = lstm.model
			
		y_pred = model.predict(
			input,
			batch_size=batch_size,
			verbose=verbose
		)

		return y_pred