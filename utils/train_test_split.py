# Function splitting train and test data
def train_test_split( data, split_factor):
	i_split = int( len( data) * split_factor)

	train_data = data[ :i_split]
	test_data = data[ i_split:]

	return train_data, test_data