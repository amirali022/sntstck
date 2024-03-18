import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

# groupby helper
def groupby_func( idx, lr, l):
	if not np.isnan( lr[ "Adj Close"][ idx]):
		return lr[ "Date"][ idx]
	
	d = lr[ "Date"][ idx]

	FORMAT = "%Y-%m-%d"

	while not ( l[ "Date"] == d).any():
		if d > l[ "Date"].max():
			return None
		d = datetime.strptime( d, FORMAT) + timedelta( days=1)
		d = datetime.strftime( d, FORMAT)

	return d


# Function for merging historical data with sentiment scores
def merge_historical_with_sentiment( historical, sentiment):
	lr = pd.merge(
		left=historical,
		right=sentiment,
		on="Date",
		how="outer",
		sort=True
	)

	merged = lr.fillna( value={ "neg": 0, "neu": 0, "pos": 0, "compound": 0}).groupby( lambda i: groupby_func( i, lr, historical)).mean( numeric_only=True)

	new_df = pd.DataFrame(
		data={
			"Open": merged[ "Open"],
			"High": merged[ "High"],
			"Low": merged[ "Low"],
			"Close": merged[ "Close"],
			"Adj Close": merged[ "Adj Close"],
			"Volume": merged[ "Volume"],
			"Neg": merged[ "neg"],
			"Neu": merged[ "neu"],
			"Pos": merged[ "pos"],
			"Compound": merged[ "compound"]
		}
	)
	new_df.index = merged.index
	new_df.index.name = "Date"

	return new_df