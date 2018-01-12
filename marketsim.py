# Vince Petaccio

import pandas as pd
import numpy as np
from util import get_data
import math


def compute_portvals(orders_file="./orders/orders.csv", start_val=100000, max_leverage=2.0, as_frame=False):
    """Compute the daily values of the portfolio defined by the orders file"""

    if as_frame:
        orders = orders_file
    else:
        # First thing's first. Read in the orders file.
        orders = pd.read_csv(orders_file, index_col=0, parse_dates=True, header=0)

    # Make the list of symbols ordered
    symbols = orders['Symbol'].unique().tolist()

    # Grab the data from the get_data method. Keep the SPY data to use SPY for filtering out non-trading days
    prices = get_data(symbols, pd.date_range(orders.index[0], orders.index[-1]), colname='Adj Close')
    if 'SPY' not in symbols:
        prices = prices.drop('SPY', 1)  # remove SPY if it wasn't traded (presumably as an index fund investment)

    if orders.index[0] != prices.index[0]:
        orders.index = prices.index

    # Add a CASH column to prices
    prices = prices.join(pd.DataFrame(np.ones(len(prices.index)), index=prices.index, columns=['CASH', ]))

    # Create the trades dataframe of 0s, with the same indices and columns as prices
    trades = pd.DataFrame(np.zeros((len(prices.index), len(prices.columns))), index=prices.index,
                          columns=prices.columns)

    # Run the orders to fill in the trades frame
    trades = place_orders(orders, prices, trades, start_val, max_leverage)

    # Make the holdings dataframe
    holdings = trades
    holdings['CASH'][0] += start_val  # Add the starting cash value to the first row in holdings
    holdings = holdings.cumsum()  # Convert holdings into a cumulative sum

    # Make the values dataframe
    values = holdings*prices

    # Sum along rows to get daily values
    portvals = values.sum(1)

    return portvals


def place_orders(orders, prices, trades, start_val, max_leverage):
    """Parses through the orders and updates the balance of each position"""
    order_dates = orders.index.unique()  # Get a list of the dates on which orders were placed
    order_number = 0
    # Iterate through all of the orders, but group them by day so that overleveraged orders are purged along with ALL
    # orders for that day
    for order_date in order_dates:
        temp_df = orders.ix[order_date]  # Make a temporary frame for this date's orders
        temp_trades = trades.copy(deep=True)  # Make temporary trades frame to check leveraging
        if len(temp_df.shape) == 1:
            orders_today = 1
        else:
            orders_today = len(temp_df.index)
        for order in range(0, orders_today):
            equity = orders['Symbol'][order_number]  # Index numerically since date is the same for all daily orders
            shares = int(orders['Shares'][order_number])
            if orders['Order'][order_number] == 'BUY':  # Buy order
                temp_trades['CASH'][order_date] += -prices[equity][order_date] * shares
                temp_trades[equity][order_date] += shares
            elif orders['Order'][order_number] == 'SELL':  # Sell order
                temp_trades['CASH'][order_date] += prices[equity][order_date] * shares
                temp_trades[equity][order_date] += -shares
            order_number += 1
        if lever_check(prices, temp_trades, order_date, start_val, max_leverage):
            trades = temp_trades.copy(deep=True)
    return trades


def lever_check(prices, lev_trades, today, start_val, max_leverage):
    """Checks whether trades on a day are over-leveraged. """
    if max_leverage == -1: return True
    held = lev_trades.copy(deep=True)
    held['CASH'][0] += start_val
    held = held.cumsum()
    positions = held*prices
    cash = positions['CASH'][today]
    positions = positions.drop('CASH', 1)
    numerator = (abs(positions)).sum(1)[today]
    denominator = positions.sum(1)[today] + cash
    leverage = numerator/denominator
    return leverage < max_leverage


def author():
    """(^^,)"""
    return 'vpetaccio3'


def assess_portfolio(portvals, rfr=0.0, sf=252.0):
    """Computers portfolio statistics"""
    daily_returns = (portvals/portvals.shift(1) - 1)[1:]
    cum_returns = portvals[-1]/portvals[0] - 1
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    k = math.sqrt(sf)
    daily_rfr = ((1.0 + rfr)**(1/sf)) - 1
    sharpe_ratio = (k*((daily_returns - daily_rfr).mean()))/std_daily_return
    return cum_returns, avg_daily_return, std_daily_return, sharpe_ratio