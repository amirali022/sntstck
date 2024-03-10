import keras

# Mean Squared Error (MSE)
def MSE( y_true, y_pred):
	metric = keras.metrics.MeanSquaredError()
	metric.update_state( y_true, y_pred)
	return metric.result().numpy()

# Root Mean Squared Error (RMSE)
def RMSE( y_true, y_pred):
	metric = keras.metrics.RootMeanSquaredError()
	metric.update_state( y_true, y_pred)
	return metric.result().numpy()

# Mean Absolute Error (MAE)
def MAE( y_true, y_pred):
	metric = keras.metrics.MeanAbsoluteError()
	metric.update_state( y_true, y_pred)
	return metric.result().numpy()

# Mean Absolute Percentage Error (MAPE)
def MAPE( y_true, y_pred):
	metric = keras.metrics.MeanAbsolutePercentageError()
	metric.update_state( y_true, y_pred)
	return metric.result().numpy()

def evaluate( y_true, y_pred):
	mse = MSE( y_true, y_pred)
	rmse = RMSE( y_true, y_pred)
	mae = MAE( y_true, y_pred)
	mape = MAPE( y_true, y_pred)

	print( f"MSE: { mse:.2f}\nRMSE: { rmse:.2f}\nMAE: { mae:.2f}\nMAPE: { mape:.2f}%")