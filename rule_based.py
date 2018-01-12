# Vince Petaccio

import numpy as np
import pandas as pd
from indicators import compute_sma, compute_bollinger, compute_aroon


def trade_by_hand(prices, swindow, bwindow, awindow, hold_for, sma_lims,
                  bol_lims, aroon_lims, block_size, symbol):
    # First get the SMA data and isolate the ratio values
    sma_ratio = compute_sma(prices, swindow, as_ratio=True).ix[:, 0].dropna()
    sma_overbought = sma_ratio[sma_ratio > sma_lims[0]]
    sma_oversold = sma_ratio[sma_ratio < sma_lims[1]]
    # Repeat for Bollinger Band Percentage
    boll = compute_bollinger(prices, bwindow, as_indicator=True).ix[:, 0].dropna()
    boll_overbought = boll[boll > bol_lims[0]]
    boll_oversold = boll[boll < bol_lims[1]]
    # Join the frames so far
    overbought = sma_overbought.to_frame().join(boll_overbought.to_frame(), how='inner')
    oversold = sma_oversold.to_frame().join(boll_oversold.to_frame(), how='inner')
    # Finally, get the Aroon oscillator values
    aroon = compute_aroon(prices, awindow, as_indicator=True).ix[:, 0].dropna() / 100
    aroon_overbought = aroon[aroon < aroon_lims[0]]
    aroon_oversold = aroon[aroon > aroon_lims[1]]
    # Add the Aroon values to the dataframes
    overbought = overbought.join(aroon_overbought.to_frame(), how='inner')
    oversold = oversold.join(aroon_oversold.to_frame(), how='inner')
    return make_orders(symbol, overbought, oversold, block_size, hold_for)


def make_orders(symbol, overbought, oversold, block_size, hold_for):
    position = 0
    orders = pd.DataFrame(columns=['Symbol', 'Order', 'Shares'])
    overbought['marker'] = np.zeros(len(overbought)).astype(int)
    oversold['marker'] = np.ones(len(oversold)).astype(int)
    signals = pd.concat([overbought['marker'], oversold['marker']]).sort_index()
    i = -1
    hold = pd.Timedelta(hold_for, unit='d')
    last_trade = signals.index[0] - pd.Timedelta(hold_for + 1, unit='d')
    for signal in signals:
        i += 1
        today = signals.index[i]
        if today >= last_trade + hold:
            if signal:
                if position < block_size:
                    position += block_size
                    new_order = pd.DataFrame({'Symbol': pd.Series(symbol, index=[today]),
                                              'Order': pd.Series('BUY', index=[today]),
                                              'Shares': pd.Series(block_size, index=[today])})
                    orders = orders.append(new_order)
                    last_trade = today
            else:
                if position > -block_size:
                    position -= block_size
                    new_order = pd.DataFrame({'Symbol': pd.Series(symbol, index=[today]),
                                              'Order': pd.Series('SELL', index=[today]),
                                              'Shares': pd.Series(block_size, index=[today])})
                    orders = orders.append(new_order)
                    last_trade = today
    return orders
