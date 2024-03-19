from sklearn.svm import SVR

def reshaper( data):
	n_samples, windows, dims = data.shape

	return data.reshape( ( n_samples, windows * dims))

# Epsilon-Support Vector Regression
class SupportVectorRegressor:
	def __init__( self):
		model = SVR()

		self.model = model

	def fit( self, input, label):

		self.model.fit( reshaper( input), label)

	def predict( self, input):
		y_pred = self.model.predict( reshaper( input))

		return y_pred