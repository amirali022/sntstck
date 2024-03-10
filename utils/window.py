import numpy as np

# Function to create window data
def create_win_data( data, seq_len):
	len_data = len( data)

	data_windows = []

	for i in range( len_data - seq_len):
		data_windows.append( data[ i:i + seq_len])

	return np.array( data_windows).astype( float)