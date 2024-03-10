import numpy as np

# Function for normalizing data
def normalize( data_windows):
	normalized_data = []
	record_min = []
	record_max = []

	for i in range( len( data_windows)):
		temp = data_windows[ i]

		minimum = temp.min( axis=0)

		record_min.append( minimum[ -1])

		maximum = temp.max( axis=0)

		record_max.append( maximum[ -1])

		diff = maximum - minimum

		temp = temp - minimum

		temp = temp / diff

		normalized_data.append( temp)

	normalized_data = np.array( normalized_data)
	record_min = np.array( record_min)
	record_max = np.array( record_max)

	return normalized_data, record_min, record_max

# Function for de-normalizing data
def denormalize( data, record_min, record_max):
	denormalized_data = []

	for i in range( len( data)):
		minimum = record_min[ i]
		maximum = record_max[ i]
		diff = maximum - minimum
		denormalized_data.append( data[ i] * diff + minimum)

	return np.array( denormalized_data)