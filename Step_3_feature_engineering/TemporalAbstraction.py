import numpy as np
import pandas as pd
import scipy.stats as stats
from warnings import simplefilter

# Class to abstract a history of numerical values we can use as an attribute.
class NumericalAbstraction:
    # pandas concat method doesn't work proerply yet so supress warning (see https://github.com/twopirllc/pandas-ta/issues/340)
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning) 
    # For the slope we need a bit more work compared to other metrics.
    # We create time points, assuming discrete time steps with fixed delta t:
    def get_slope(self, data):      
        times = np.array(range(0, len(data.index)))
        data = data.astype(np.float32)

        # Check for NaN's
        mask = ~np.isnan(data)

        # If we have no data but NaN we return NaN.
        if (len(data[mask]) == 0):
            return np.nan
        # Otherwise we return the slope.
        else:
            slope, _, _, _, _ = stats.linregress(times[mask], data[mask])
            return slope

    # This function aggregates a list of values using the specified aggregation
    # function (which can be 'mean', 'max', 'min', 'median', 'std', 'slope')
    def aggregate_value(self,data, window_size, aggregation_function):
        window = str(window_size) + 's'
        # Compute the values and return the result.
        if aggregation_function == 'mean':
            return data.rolling(window, min_periods=window_size).mean()
        elif aggregation_function == 'max':
            return data.rolling(window, min_periods=window_size).max()
        elif aggregation_function == 'min':
            return data.rolling(window, min_periods=window_size).min()
        elif aggregation_function == 'median':
            return data.rolling(window, min_periods=window_size).median()
        elif aggregation_function == 'std':
            return data.rolling(window, min_periods=window_size).std()
        elif aggregation_function == 'slope':
            return data.rolling(window, min_periods=window_size).apply(self.get_slope)
        else:
            return np.nan

    def abstract_numerical(self, data_table, cols, window_size, aggregation_function_name):   
        for agg in aggregation_function_name:
            for col in cols:                       
                data_table[col + '_temp_' + agg + '_ws_' + str(window_size)] = self.aggregate_value(data_table[col], window_size, agg)
        return data_table