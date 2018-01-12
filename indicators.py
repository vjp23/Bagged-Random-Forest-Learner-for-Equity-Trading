# Vince Petaccio

# Implement technical indicators and create plots.

# Import libraries
import pandas as pd
from pandas.stats import moments

def compute_sma(prices, sma_window, as_ratio=False, normalize=False, standardize=False):
    """
    Compute SMA (simple moving average), or SMA / price ratio
    :param prices: price data 
    :param window: SMA time span
    :param normalize: Boolean, whether to normalize price data before computation
    :param standardize: Boolean, whether to standardize returned indicator data (make z-scores)
    :return: returns a pandas dataframe of the same size as prices, indicating the SMA or SMA/price ratio
    """
    if normalize: prices /= prices.ix[-1]
    sma = prices.rolling(window=sma_window).mean()
    new_name = str(sma_window) + '-Day SMA'
    if as_ratio:
        sma = prices / sma
        new_name = str(sma_window) + '-Day Price / SMA'
    if standardize: sma = standardize_series(sma)
    sma.columns = [new_name]
    return  sma

def compute_bollinger(prices, bollinger_window, normalize=False, standardize=False, as_indicator=False):
    """
    Computes bollinger bands for bollinger_window-day rolling average
    :param prices: price data
    :param bollinger_window: SMA time span
    :param normalize: Boolean, whether to normalize price data before computation
    :param standardize: Boolean, whether to standardize returned indicator data (make z-scores)
    :param as_indicator: Boolean, whether to combine bands into a indicator value
    :return: 
    """
    if normalize: prices /= prices.ix[-1]
    standard_dev = prices.rolling(window=bollinger_window).std()
    rolling_mean = compute_sma(prices, bollinger_window)
    rolling_mean.columns = standard_dev.columns
    lower = rolling_mean - 2 * standard_dev
    upper = rolling_mean + 2 * standard_dev
    if as_indicator:
        # Combine the upper and lower bands into a single indicator value, bollinger band percentage
        bollinger_bands = (prices - lower) / (upper - lower)
        bollinger_bands.columns = ['Bollinger Band Percentage']
        if standardize:
            # Standardize the indicator
            bollinger_bands = standardize_series(bollinger_bands)
    else:
        # New dataframe with both bands
        bollinger_bands = pd.concat([lower, upper, rolling_mean], axis=1)
        bollinger_bands.columns = ['Lower Band', 'Upper Band', str(bollinger_window) + '-Day Rolling Mean']
    return bollinger_bands

def compute_aroon(prices, aroon_window, normalize=False, standardize=False, as_indicator=False):
    """
    Computers aroon oscillator for aroon_window timespan
    :param prices: price data
    :param aroon_window: Aroon periods
    :param normalize: Boolean, whether to normalize price data before computation
    :param standardize: Boolean, whether to standardize returned indicator data (make z-scores)
    :return: 
    """
    if normalize: prices /= prices.ix[-1]
    # Use rolling apply to find days since window max and min. Convert to %
    aroon_up = 100 * prices.rolling(center=False, window=aroon_window+1).apply(func = lambda x: x.argmax()) / \
               aroon_window
    aroon_down = 100 * prices.rolling(center=False, window=aroon_window+1).apply(func = lambda x: x.argmin()) / \
                 aroon_window
    aroon_osc = aroon_up - aroon_down
    if standardize:
        aroon_up = standardize_series(aroon_up)
        aroon_down = standardize_series(aroon_down)
        aroon_osc = standardize_series(aroon_osc)
    if as_indicator:
        aroon_osc.columns = ['Aroon Oscillator']
        return aroon_osc
    all_aroon = pd.concat([aroon_up, aroon_down, aroon_osc], axis=1)
    all_aroon.columns = ['Aroon Up', 'Aroon Down', 'Aroon Oscillator']
    return all_aroon

def standardize_series(data_series):
    return (data_series - data_series.mean()) / data_series.std()