# Vince Petaccio

# Import libraries and local files
import numpy as np
import pandas as pd
from indicators import compute_sma, compute_bollinger, compute_aroon
import RTLearnerClassifier as rtc
import BagLearner as bl


def trade_by_RTL(prices, swindow, bwindow, awindow, hold_for, block_size, symbol, long_at, short_at, leaf, bags):
    # Get indicators
    sma_ratio = compute_sma(prices, swindow, as_ratio=True, standardize=True).ix[:, 0].dropna()
    boll = compute_bollinger(prices, bwindow, as_indicator=True, standardize=True).ix[:, 0].dropna()
    aroon = compute_aroon(prices, awindow, as_indicator=True, standardize=True).ix[:, 0].dropna()
    return make_orders(symbol, prices, sma_ratio, boll, aroon, block_size, hold_for, long_at, short_at, leaf, bags)


def make_orders(symbol, prices, sma_ratio, boll, aroon, block_size, hold_for, long_at, short_at, leaf, bags):
    if bags > 0:
        learner = bl.BagLearner(rtc.RTLearnerClassifier, {"leaf_size": leaf}, bags)
    else: learner = rtc.RTLearnerClassifier(leaf_size=leaf)
    # Compute returns at end of hold_for-day period
    returns = prices.shift(-hold_for) - prices
    returns[returns[symbol[0]] > long_at] = 1
    returns[returns[symbol[0]] < short_at] = -1
    returns[(returns[symbol[0]] <= long_at) & (returns[symbol[0]] >= short_at)] = 0
    classified_returns = returns.dropna()
    # Build the training data
    training_data = pd.concat([sma_ratio, boll, aroon, classified_returns], axis=1).dropna()
    training_matrix = training_data.as_matrix()
    training_x = training_matrix[:, 0:-1]
    training_y = training_matrix[:, -1]
    # Train the learner
    learner.addEvidence(training_x, training_y)
    # Query the learner with in-sample data
    learner_orders = learner.query(training_x)
    # Get rid of orders that violate rules
    position = 0
    orders = pd.DataFrame(columns=['Symbol', 'Order', 'Shares'])
    hold = pd.Timedelta(hold_for, unit='d')
    last_trade = classified_returns.index[0] - pd.Timedelta(hold_for + 1, unit='d')
    for i in range(0, len(learner_orders)):
        today = training_data.index[i]
        if today >= last_trade + hold: # Enough time has passed to permit another trade
            if learner_orders[i] == 1: # Buy signal
                if position < block_size: # Another buy permitted
                    position += block_size
                    new_order = pd.DataFrame({'Symbol': pd.Series(symbol, index=[today]),
                                              'Order': pd.Series('BUY', index=[today]),
                                              'Shares': pd.Series(block_size, index=[today])})
                    orders = orders.append(new_order)
                    last_trade = today
            elif learner_orders[i] == -1: # Sell signal
                if position > -block_size: # Another sell permitted
                    position -= block_size
                    new_order = pd.DataFrame({'Symbol': pd.Series(symbol, index=[today]),
                                              'Order': pd.Series('SELL', index=[today]),
                                              'Shares': pd.Series(block_size, index=[today])})
                    orders = orders.append(new_order)
                    last_trade = today
    return orders
