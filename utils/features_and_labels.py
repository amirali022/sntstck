# Function for obtaining features and labels
def features_and_labels( data):
	x = data[ :, :-1]
	y = data[ :, -1, -1]

	return x, y