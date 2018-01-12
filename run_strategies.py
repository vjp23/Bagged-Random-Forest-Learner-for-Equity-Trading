# Vince Petaccio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# Import other local files
from indicators import compute_sma, compute_bollinger, compute_aroon
from util import get_data
from marketsim import compute_portvals, assess_portfolio
from rule_based import trade_by_hand
from ML_based import trade_by_RTL
import RTLearnerClassifier as rtl
import BagLearner as bl

# Switches for choosing which code to run ------------------------------------------------------------------------------
run_indicators = False
run_best = False
run_rule_based = False                  #                                _      _
run_ML_based = False                    #                               |_|    |_|
run_visualization = False               #                              _    ^     _
run_comparative = False                 #                               -________-
save_figs = False
show_figs = False
# Set variables --------------------------------------------------------------------------------------------------------
symbol = ['AAPL']
hold_for = 21
block_size = 200
starting_cash = 100000

sma_window = 20
bollinger_window = 20
aroon_window = 20

sma_fences = [1.02, 0.98]
bollinger_fences = [0.7, 0.4]
aroon_fences = [-0.5, 0.5]

long_at = 0.005
short_at = -0.005
leaf_size = 6
bagged_learners = 50

training_start_date = dt.datetime(2008, 1, 1)
training_end_date = dt.datetime(2009, 12, 31)

test_start_date = dt.datetime(2010, 1, 1)
test_end_date = dt.datetime(2011, 12, 31)

figure_size = (18, 10)
plot_title_size = 26
plot_axis_label_size = 21
# ----------------------------------------------------------------------------------------------------------------------

# Setup the date ranges, get price data, etc
training_dates = pd.date_range(training_start_date, training_end_date)
test_dates = pd.date_range(test_start_date, test_end_date)
all_training_prices = get_data(symbol, training_dates)
all_test_prices = get_data(symbol, test_dates)
training_prices = all_training_prices[symbol]
test_prices = all_test_prices[symbol]
training_norm = training_prices / training_prices.ix[0]
test_norm = test_prices / test_prices.ix[0]
training_SPY = all_training_prices['SPY']
test_SPY = all_test_prices['SPY']

if run_indicators:
    # First do SMA / Price ratio
    train_sma = compute_sma(training_norm, sma_window)
    train_sma_ratio = compute_sma(training_norm, sma_window, as_ratio=True)
    # Combine the SMA data into a single dataframe
    sma = pd.concat([training_norm, train_sma, train_sma_ratio], axis = 1)
    # Plot the SMA data
    fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios':[2, 1]}, figsize=(figure_size[0], figure_size[1] + 3))
    sma[symbol].plot(ax=axes[0], color='#673AB7', sharex=axes[1], alpha=1.0)
    sma[sma.columns[1]].plot(ax=axes[0], color='#E91E63', sharex=axes[1], alpha=0.6)
    axes[1].hlines(y=1.0, xmin=sma.index[0], xmax=sma.index[-1], alpha=0.8, colors='black', linewidth=0.2)
    sma[sma.columns[2]].plot(ax=axes[1], color='#03A9F4')
    axes[0].set_title(str(sma_window) + '-Day Simple Moving Average (SMA)',
                      fontname='Helvetica Neue LT Std', fontsize=plot_title_size)
    axes[0].set_ylabel('Normalized Price', fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
    axes[0].legend([symbol[0], str(sma_window) + '-Day SMA'])
    axes[1].set_title(str(sma_window) + '-Day Price / SMA Ratio',
                      fontname='Helvetica Neue LT Std', fontsize=plot_title_size)
    axes[1].set_ylabel('Price / SMA', fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
    if save_figs:   fig.savefig("{}_SMA.png".format(str(symbol[0])), dpi=100, bbox_inches='tight')

    # Next do Bollinger Bands
    bollinger = compute_bollinger(training_norm, bollinger_window)
    bollinger_perc = compute_bollinger(training_norm, bollinger_window, False, False, True)
    # Plot it
    bol_fig, bol_axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},
                                     figsize=(figure_size[0], figure_size[1] + 3))
    training_norm.plot(ax=bol_axes[0], color='black', sharex=bol_axes[1])
    bollinger.plot(ax=bol_axes[0], color=['#0277BD', '#40C4FF', '#757575'], alpha=0.5, sharex=bol_axes[1])
    bol_axes[0].set_title(str(bollinger_window) + '-Day 2-' + r'$\sigma$' + ' Bollinger Bands',
                 fontname='Helvetica Neue LT Std', fontsize=plot_title_size)
    bol_axes[0].set_ylabel('Normalized Price', fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
    bollinger_perc.plot(ax=bol_axes[1], color='#0DFFBC', legend=None)
    bol_axes[1].set_title(str(bollinger_window) + '-Day Bollinger Band Percentage', fontname='Helvetica Neue LT Std',
                          fontsize=plot_title_size)
    bol_axes[1].set_ylabel('Bollinger Band Percentage', fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
    bol_axes[1].hlines(y=0.0, xmin=training_norm.index[0],
                       xmax=training_norm.index[-1], alpha=0.8, colors='#00CDFF', linewidth=0.4)
    bol_axes[1].hlines(y=1.0, xmin=training_norm.index[0],
                       xmax=training_norm.index[-1], alpha=0.8, colors='#0DC9FF', linewidth=0.4)
    if save_figs:   bol_fig.savefig("{}_Bollinger.png".format(str(symbol[0])), dpi=100, bbox_inches='tight')

    # Finally, do Aroon Indicator
    aroon = compute_aroon(training_norm, aroon_window, standardize=True)
    # Plot it!
    aroon_fig, aroon_axes = plt.subplots(3, 1, gridspec_kw={'height_ratios':[3, 1, 1.2]},
                                         figsize=(figure_size[0], figure_size[1] + 7))
    training_norm.plot(ax=aroon_axes[0], color='black', legend=None)
    aroon['Aroon Up'].plot(ax=aroon_axes[1], color='#1DE9B6', sharex=aroon_axes[2], alpha=0.8)
    aroon['Aroon Down'].plot(ax=aroon_axes[1], color='#4FC3F7', sharex=aroon_axes[2], alpha=0.8)
    aroon['Aroon Oscillator'].plot(ax=aroon_axes[2], color='#9C27B0', alpha=0.8)
    aroon_axes[0].set_title(symbol[0], fontname='Helvetica Neue LT Std', fontsize=plot_title_size)
    aroon_axes[0].set_ylabel('Normalized Price', fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
    aroon_axes[1].set_title(str(aroon_window) + '-Day Aroon Indicators',
                            fontname='Helvetica Neue LT Std', fontsize=plot_title_size)
    aroon_axes[1].set_ylabel('Aroon Indicators', fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
    aroon_axes[1].legend(['Aroon Up', 'Aroon Down'])
    aroon_axes[2].set_title(str(aroon_window) + '-Day Aroon Oscillator',
                            fontname='Helvetica Neue LT Std', fontsize=plot_title_size)
    aroon_axes[2].set_ylabel('Aroon Oscillator', fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
    if save_figs:   aroon_fig.savefig("{}_Aroon.png".format(str(symbol[0])), dpi=100, bbox_inches='tight')

if run_best:
    # For the benchmark portfolio, just use the normalized prices
    # First get the benchmark and price data
    today = training_norm.index[0]
    last_day = training_norm.index[-1]
    bench_trades = pd.DataFrame({'Symbol': pd.Series(symbol, index=[today]),
                                 'Order': pd.Series('BUY', index=[today]),
                                 'Shares': pd.Series(block_size, index=[today])})
    bench_trades = bench_trades.append(pd.DataFrame({'Symbol': pd.Series(symbol, index=[last_day]),
                                                     'Order': pd.Series('SELL', index=[last_day]),
                                                     'Shares': pd.Series(block_size, index=[last_day])}))
    benchmark = compute_portvals(bench_trades, starting_cash, -1, 1)
    benchmark /= benchmark[0]
    benchmark.columns = symbol

    # Simulate the best possible portfolio by buying when tomorrow's returns will be positive, and sell when negative
    returns_tmrw = training_norm.shift(-1) - training_norm
    returns_tmrw = returns_tmrw[:-1]
    best_possible_orders = pd.DataFrame(columns=['Symbol', 'Order', 'Shares'])
    last_trans = -1
    for i in range(0, len(returns_tmrw.index) - 1):
        ordered = 0
        today = returns_tmrw.index[i]
        if returns_tmrw[symbol[0]][i] >= 0 and last_trans == -1:
            new_order = pd.DataFrame({'Symbol': pd.Series(symbol, index=[today]),
                                     'Order': pd.Series('BUY', index=[today]),
                                     'Shares': pd.Series(block_size, index=[today])})
            last_trans = 1
            ordered = 1
        elif returns_tmrw[symbol[0]][i] < 0 and last_trans == 1:
            new_order = pd.DataFrame({'Symbol': pd.Series(symbol, index=[today]),
                                     'Order': pd.Series('SELL', index=[today]),
                                     'Shares': pd.Series(block_size, index=[today])})
            last_trans = -1
            ordered = 1
        if bool(ordered):
            best_possible_orders = best_possible_orders.append(new_order)
    if last_trans == 1:
        today = returns_tmrw.index[-1]
        best_possible_orders = best_possible_orders.append(pd.DataFrame({'Symbol': pd.Series(symbol, index=[today]),
                                                                         'Order': pd.Series('SELL', index=[today]),
                                                                         'Shares': pd.Series(block_size,
                                                                                             index=[today])}))
    best_possible_portvals = compute_portvals(best_possible_orders, starting_cash, -1, 1)
    best_possible_portvals /= best_possible_portvals[0] # Normalize the best possible portfolio's values
    best_possible_portvals.columns = [symbol]

    # Get statistics for the two strategies
    bench_cum_ret, bench_avg_daily_ret, bench_std_daily_ret, bench_sharpe_ratio = assess_portfolio(benchmark.ix[:, 0])
    best_cum_ret, best_avg_daily_ret, best_std_daily_ret, best_sharpe_ratio = assess_portfolio(best_possible_portvals)
    # Print the statistics of the two portfolios
    print "Cumulative Return of Benchmark: {}".format(bench_cum_ret)
    print "Cumulative Return of Best Possible Strategy: {}".format(best_cum_ret)
    print
    print "Standard Deviation of Benchmark: {}".format(bench_std_daily_ret)
    print "Standard Deviation of Best Possible Strategy: {}".format(best_std_daily_ret)
    print
    print "Average Daily Return of Benchmark: {}".format(bench_avg_daily_ret)
    print "Average Daily Return of Best Possible Strategy: {}".format(best_avg_daily_ret)
    # Plot the figures
    best_ax = benchmark.plot(color='black', figsize=figure_size)
    best_possible_portvals.plot(ax=best_ax, color='blue')
    best_ax.set_title('Benchmark and Best Possible Portfolio Values',
                      fontname='Helvetica Neue LT Std', fontsize=plot_title_size)
    best_ax.set_ylabel('Normalized Portfolio Value', fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
    plt.legend(labels=['Benchmark Portfolio', 'Best Possible Portfolio'])
    if save_figs:
        best_fig = plt.gcf()
        best_fig.savefig("{}_Best_Strategy.png".format(str(symbol[0])), dpi=100, bbox_inches='tight')

if run_rule_based or run_ML_based:
    # Build the list of manual trades
    manual_trades = trade_by_hand(training_norm, sma_window, bollinger_window, aroon_window, hold_for,
                                           sma_fences, bollinger_fences, aroon_fences, block_size, symbol)
    # Get the normalized portfolio value for the manual trades
    manual_portvals = compute_portvals(manual_trades, starting_cash, -1, 1)
    manual_portvals /= manual_portvals[0]
    manual_portvals.columns = symbol
    # For the benchmark portfolio, just use the normalized prices
    # First get the benchmark and price data
    today = training_norm.index[0]
    last_day = training_norm.index[-1]
    bench_trades = pd.DataFrame({'Symbol': pd.Series(symbol, index=[today]),
                                 'Order': pd.Series('BUY', index=[today]),
                                 'Shares': pd.Series(block_size, index=[today])})
    bench_trades = bench_trades.append(pd.DataFrame({'Symbol': pd.Series(symbol, index=[last_day]),
                                                     'Order': pd.Series('SELL', index=[last_day]),
                                                     'Shares': pd.Series(block_size, index=[last_day])}))
    benchmark = compute_portvals(bench_trades, starting_cash, -1, 1)
    benchmark /= benchmark[0]
    benchmark.columns = symbol
    # Pad the manual_portvals dataframe appropriately
    hesitated = pd.date_range(benchmark.index[0], manual_portvals.index[0] - pd.Timedelta('1 day'))
    quit = pd.date_range(manual_portvals.index[-1] + pd.Timedelta('1 day'), benchmark.index[-1])
    start = pd.DataFrame(np.ones(len(hesitated)).astype(int), index = hesitated)
    stop = pd.DataFrame(manual_portvals[-1] * np.ones(len(quit)), index = quit)
    manual_portvals = pd.concat([manual_portvals, start, stop]).sort_index()
    if run_rule_based:
        # Plot it!!!!!!!!!
        man_axe = benchmark.plot(color='black', figsize=figure_size)
        manual_portvals.plot(ax=man_axe, color='#00CDFF')
        # Plot the short lines and the buy lines
        short_lines = []
        long_lines = []
        position = 0
        for j in range(0, len(manual_trades['Order'])):
            if manual_trades['Order'][j] == 'SELL': position -= 1
            else: position += 1
            if position < 0: short_lines.append(manual_trades.index[j])
            elif position > 0: long_lines.append(manual_trades.index[j])
        for s in short_lines: man_axe.axvline(x=s, alpha=0.5, color='#76FF03', linewidth=0.8)
        for b in long_lines: man_axe.axvline(x=b, alpha=0.5, color='#FF5722', linewidth=0.8)
        man_axe.set_title('Manual Strategy vs Benchmark', fontname='Helvetica Neue LT Std', fontsize=plot_title_size)
        man_axe.set_ylabel('Normalized Portfolio Value', fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
        man_axe.legend(['Benchmark', 'Manual Strategy'])
        if save_figs:
            man_fig = plt.gcf()
            man_fig.savefig("{}_Manual_Strategy.png".format(str(symbol[0])), dpi=100, bbox_inches='tight')

if run_ML_based:
    # Train the learner and use it to build a valid order file
    ml_trades = trade_by_RTL(training_norm, sma_window, bollinger_window, aroon_window,
                             hold_for, block_size, symbol, long_at, short_at, leaf_size, bagged_learners)
    # Run the trades determined by the learner
    ml_portvals = compute_portvals(ml_trades, starting_cash, -1, 1)
    ml_portvals /= ml_portvals[0]
    ml_portvals.columns = symbol
    # Pad the manual_portvals dataframe appropriately
    hesitated_ml = pd.date_range(benchmark.index[0], ml_portvals.index[0] - pd.Timedelta('1 day'))
    quit_ml = pd.date_range(ml_portvals.index[-1] + pd.Timedelta('1 day'), benchmark.index[-1])
    start_ml = pd.DataFrame(np.ones(len(hesitated_ml)).astype(int), index=hesitated_ml)
    stop_ml = pd.DataFrame(ml_portvals[-1] * np.ones(len(quit_ml)), index=quit_ml)
    ml_portvals = pd.concat([ml_portvals, start_ml, stop_ml]).sort_index()
    # Plawdit
    ml_ax = benchmark.plot(color='black', figsize=figure_size)
    manual_portvals.plot(ax=ml_ax, color='#00CDFF')
    ml_portvals.plot(ax=ml_ax, color='#0DFFBC')
    ml_ax.legend(['Benchmark', 'Manual Strategy', 'Machine Learning Strategy'])
    # Plot the short lines and the buy lines
    ml_short_lines = []
    ml_long_lines = []
    position = 0
    for k in range(0, len(ml_trades['Order'])):
        if ml_trades['Order'][k] == 'SELL':
            position -= 1
        else:
            position += 1
        if position < 0:
            ml_short_lines.append(ml_trades.index[k])
        elif position > 0:
            ml_long_lines.append(ml_trades.index[k])
    for sml in ml_short_lines: ml_ax.axvline(x=sml, alpha=0.5, color='#76FF03', linewidth=1.1)
    for bml in ml_long_lines: ml_ax.axvline(x=bml, alpha=0.5, color='#FF5722', linewidth=0.8)
    ml_ax.set_title('Machine Learning Strategy vs Manual Strategy',
                    fontname='Helvetica Neue LT Std', fontsize=plot_title_size)
    ml_ax.set_ylabel('Normalized Portfolio Value', fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
    if save_figs:
        ml_fig = plt.gcf()
        ml_fig.savefig("{}_ML_Strategy.png".format(str(symbol[0])), dpi=100, bbox_inches='tight')

if run_visualization or run_comparative:
    # Get indicators
    prices = training_norm
    sma_rat_raw = compute_sma(prices, sma_window, as_ratio=True).ix[:, 0].dropna()
    sma_ratio = compute_sma(prices, sma_window, as_ratio=True, standardize=True).ix[:, 0].dropna()
    boll_raw = compute_bollinger(prices, bollinger_window, as_indicator=True).ix[:, 0].dropna()
    boll = compute_bollinger(prices, bollinger_window, as_indicator=True, standardize=True).ix[:, 0].dropna()
    aroon_raw = compute_aroon(prices, aroon_window, as_indicator=True).ix[:, 0].dropna()
    aroon = compute_aroon(prices, aroon_window, as_indicator=True, standardize=True).ix[:, 0].dropna()
    x1 = sma_ratio
    x2 = aroon
    if run_visualization:
        # Get the rule-based classifications
        sf = (sma_fences - sma_rat_raw.mean()) / sma_rat_raw.std()
        bf = (bollinger_fences - boll_raw.mean()) / boll_raw.std()
        af = (aroon_fences - aroon_raw.mean()) / aroon_raw.std()
        sma_overbought = sma_ratio[sma_ratio > sf[0]]
        sma_oversold = sma_ratio[sma_ratio < sf[1]]
        boll_overbought = boll[boll > bf[0]]
        boll_oversold = boll[boll < bf[1]]
        overbought = sma_overbought.to_frame().join(boll_overbought.to_frame(), how='inner')
        oversold = sma_oversold.to_frame().join(boll_oversold.to_frame(), how='inner')
        aroon_overbought = aroon[aroon < af[0]]
        aroon_oversold = aroon[aroon > af[1]]
        overbought = overbought.join(aroon_overbought.to_frame(), how='inner')
        oversold = oversold.join(aroon_oversold.to_frame(), how='inner')
        overbought['Indicator'] = -1 * np.ones(len(overbought)).astype(int)
        oversold['Indicator'] = np.ones(len(oversold)).astype(int)
        signals = pd.concat([overbought['Indicator'], oversold['Indicator']]).sort_index()
        man = pd.DataFrame(pd.concat([x1, x2], axis=1)).dropna()
        man_sig = man.join(signals).fillna(0)
        black_sma = man_sig[man_sig['Indicator'] == 0]
        green_sma = man_sig[man_sig['Indicator'] == 1]
        red_sma = man_sig[man_sig['Indicator'] == -1]
        del black_sma['Indicator']; del green_sma['Indicator']; del red_sma['Indicator']
        xbsma = black_sma[black_sma.columns[0]]
        ybsma = black_sma[black_sma.columns[1]]
        xgsma = green_sma[green_sma.columns[0]]
        ygsma = green_sma[green_sma.columns[1]]
        xrsma = red_sma[red_sma.columns[0]]
        yrsma = red_sma[red_sma.columns[1]]
        # Plot it
        smafig = plt.figure(figsize=[figure_size[1], figure_size[1]])
        smax = smafig.add_subplot(111)
        smax.scatter(xbsma, ybsma, s=7, color='black')
        smax.scatter(xgsma, ygsma, s=7, color='#76FF03')
        smax.scatter(xrsma, yrsma, s=7, color='#FF5722')
        smax.set_xlim([-1.5, 1.5]); smax.set_ylim([-1.5, 1.5])
        smax.legend(['DO NOTHING', 'LONG', 'SHORT'])
        smax.set_title('Classification by Price / SMA Ratio and Aroon Oscillator',
                       fontname='Helvetica Neue LT Std', fontsize=plot_title_size)
        smax.set_xlabel('Standardized Price / SMA Ratio',
                        fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
        smax.set_ylabel('Standardized Aroon Oscillator',
                        fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
        if save_figs: smafig.savefig("{}_SMA_Visualization.png".format(str(symbol[0])), dpi=100, bbox_inches='tight')

    # Build the ML training classifications
    returns = prices.shift(-hold_for) - prices
    returns[returns[symbol[0]] > long_at] = 1
    returns[returns[symbol[0]] < short_at] = -1
    returns[(returns[symbol[0]] <= long_at) & (returns[symbol[0]] >= short_at)] = 0
    classified_returns = returns.dropna()
    if run_visualization:
        # Build the training data
        ml_training_data = pd.concat([sma_ratio, aroon, classified_returns], axis=1).dropna()
        ml_training_data.columns = [ml_training_data.columns[0], ml_training_data.columns[1], 'Indicator']
        black_mlt = ml_training_data[ml_training_data['Indicator'] == 0]
        green_mlt = ml_training_data[ml_training_data['Indicator'] == 1]
        red_mlt = ml_training_data[ml_training_data['Indicator'] == -1]
        del black_mlt['Indicator']; del green_mlt['Indicator']; del red_mlt['Indicator']
        xbmlt = black_mlt[black_mlt.columns[0]]
        ybmlt = black_mlt[black_mlt.columns[1]]
        xgmlt = green_mlt[green_mlt.columns[0]]
        ygmlt = green_mlt[green_mlt.columns[1]]
        xrmlt = red_mlt[red_mlt.columns[0]]
        yrmlt = red_mlt[red_mlt.columns[1]]
        # Plot it
        mltfig = plt.figure(figsize=[figure_size[1], figure_size[1]])
        mltax = mltfig.add_subplot(111)
        mltax.scatter(xbmlt, ybmlt, s=7, color='black')
        mltax.scatter(xgmlt, ygmlt, s=7, color='#76FF03')
        mltax.scatter(xrmlt, yrmlt, s=7, color='#FF5722')
        mltax.set_xlim([-1.5, 1.5]); mltax.set_ylim([-1.5, 1.5])
        mltax.legend(['DO NOTHING', 'LONG', 'SHORT'])
        mltax.set_title('Classification by Learner Training Data',
                       fontname='Helvetica Neue LT Std', fontsize=plot_title_size)
        mltax.set_xlabel('Standardized Price / SMA Ratio', fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
        mltax.set_ylabel('Standardized Aroon Oscillator', fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
        if save_figs: mltfig.savefig("{}_MLT_Visualization.png".format(str(symbol[0])), dpi=100, bbox_inches='tight')

    # Build learner classification plot
    if bagged_learners > 0:
        learner = bl.BagLearner(rtl.RTLearnerClassifier, {"leaf_size": leaf_size}, bagged_learners)
    else: learner = rtl.RTLearnerClassifier(leaf_size=leaf_size)
    # Build the training data
    training_data = pd.concat([sma_ratio, boll, aroon, classified_returns], axis=1).dropna()
    training_matrix = training_data.as_matrix()
    training_x = training_matrix[:, 0:-1]
    training_y = training_matrix[:, -1]
    # Train the learner
    learner.addEvidence(training_x, training_y)
    if run_visualization:
        # Query the learner with in-sample data
        learner_orders = learner.query(training_x)
        learner_data = np.concatenate((training_x, learner_orders[:, None]), axis=1)
        learner_data = np.delete(learner_data, 1, 1)
        learner_data = pd.DataFrame(learner_data, columns=['SMA Ratio', 'Aroon Osc', 'Indicator'])
        black_ml = learner_data[learner_data['Indicator'] == 0]
        green_ml = learner_data[learner_data['Indicator'] == 1]
        red_ml = learner_data[learner_data['Indicator'] == -1]
        del black_ml['Indicator']; del green_ml['Indicator']; del red_ml['Indicator']
        xbml = black_ml[black_ml.columns[0]]
        ybml = black_ml[black_ml.columns[1]]
        xgml = green_ml[green_ml.columns[0]]
        ygml = green_ml[green_ml.columns[1]]
        xrml = red_ml[red_ml.columns[0]]
        yrml = red_ml[red_ml.columns[1]]
        # Plot it
        mlfig = plt.figure(figsize=[figure_size[1], figure_size[1]])
        mlax = mlfig.add_subplot(111)
        mlax.scatter(xbml, ybml, s=7, color='black')
        mlax.scatter(xgml, ygml, s=7, color='#76FF03')
        mlax.scatter(xrml, yrml, s=7, color='#FF5722')
        mlax.set_xlim([-1.5, 1.5]);
        mlax.set_ylim([-1.5, 1.5])
        mlax.legend(['DO NOTHING', 'LONG', 'SHORT'])
        mlax.set_title('Classification by Learner Classification Data',
                        fontname='Helvetica Neue LT Std', fontsize=plot_title_size)
        mlax.set_xlabel('Standardized Price / SMA Ratio',
                        fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
        mlax.set_ylabel('Standardized Aroon Oscillator',
                        fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
        if save_figs: mlfig.savefig("{}_ML_Visualization.png".format(str(symbol[0])), dpi=100, bbox_inches='tight')

if run_comparative:
    # First get the benchmark and price data
    today = test_norm.index[0]
    last_day = test_norm.index[-1]
    bench_trades = pd.DataFrame({'Symbol': pd.Series(symbol, index=[today]),
                                 'Order': pd.Series('BUY', index=[today]),
                                 'Shares': pd.Series(block_size, index=[today])})
    bench_trades = bench_trades.append(pd.DataFrame({'Symbol': pd.Series(symbol, index=[last_day]),
                                                     'Order': pd.Series('SELL', index=[last_day]),
                                                     'Shares': pd.Series(block_size, index=[last_day])}))
    benchmark = compute_portvals(bench_trades, starting_cash, -1, 1)
    benchmark /= benchmark[0]
    benchmark.columns = symbol
    prices = test_norm
    # Next do manual strategy
    # Build the list of manual trades
    tmanual_trades = trade_by_hand(test_norm, sma_window, bollinger_window, aroon_window, hold_for,
                                  sma_fences, bollinger_fences, aroon_fences, block_size, symbol)
    # Get the normalized portfolio value for the manual trades
    tmanual_portvals = compute_portvals(tmanual_trades, starting_cash, -1, 1)
    tmanual_portvals /= tmanual_portvals[0]
    tmanual_portvals.columns = symbol
    # Pad the manual_portvals dataframe appropriately
    tmhesitated = pd.date_range(benchmark.index[0], tmanual_portvals.index[0] - pd.Timedelta('1 day'))
    tmquit = pd.date_range(tmanual_portvals.index[-1] + pd.Timedelta('1 day'), benchmark.index[-1])
    tmstart = pd.DataFrame(np.ones(len(tmhesitated)).astype(int), index=tmhesitated)
    tmstop = pd.DataFrame(tmanual_portvals[-1] * np.ones(len(tmquit)), index=tmquit)
    tmanual_portvals = pd.concat([tmanual_portvals, tmstart, tmstop]).sort_index()
    # Now the ML learner
    tsma = compute_sma(prices, sma_window, as_ratio=True, standardize=True).ix[:, 0].dropna()
    tboll = compute_bollinger(prices, bollinger_window, as_indicator=True, standardize=True).ix[:, 0].dropna()
    taroon = compute_aroon(prices, aroon_window, as_indicator=True, standardize=True).ix[:, 0].dropna()
    # Build the query data
    test_data = pd.concat([tsma, tboll, taroon], axis=1).dropna()
    test_matrix = test_data.as_matrix()
    # Train the learner
    learner_orders = learner.query(test_matrix)
    # Use the indicators from the learner to generate an orders file
    position = 0
    orders = pd.DataFrame(columns=['Symbol', 'Order', 'Shares'])
    hold = pd.Timedelta(hold_for, unit='d')
    last_trade = test_data.index[0] - pd.Timedelta(hold_for + 1, unit='d')
    for i in range(0, len(learner_orders)):
        today = test_data.index[i]
        if today >= last_trade + hold:  # Enough time has passed to permit another trade
            if learner_orders[i] == 1:  # Buy signal
                if position < block_size:  # Another buy permitted
                    position += block_size
                    new_order = pd.DataFrame({'Symbol': pd.Series(symbol, index=[today]),
                                              'Order': pd.Series('BUY', index=[today]),
                                              'Shares': pd.Series(block_size, index=[today])})
                    orders = orders.append(new_order)
                    last_trade = today
            elif learner_orders[i] == -1:  # Sell signal
                if position > -block_size:  # Another sell permitted
                    position -= block_size
                    new_order = pd.DataFrame({'Symbol': pd.Series(symbol, index=[today]),
                                              'Order': pd.Series('SELL', index=[today]),
                                              'Shares': pd.Series(block_size, index=[today])})
                    orders = orders.append(new_order)
                    last_trade = today
    # Run the trades determined by the learner
    tml_portvals = compute_portvals(orders, starting_cash, -1, 1)
    tml_portvals /= tml_portvals[0]
    tml_portvals.columns = symbol
    # Pad the manual_portvals dataframe appropriately
    hesitated_tml = pd.date_range(benchmark.index[0], tml_portvals.index[0] - pd.Timedelta('1 day'))
    quit_tml = pd.date_range(tml_portvals.index[-1] + pd.Timedelta('1 day'), benchmark.index[-1])
    start_tml = pd.DataFrame(np.ones(len(hesitated_tml)).astype(int), index=hesitated_tml)
    stop_tml = pd.DataFrame(tml_portvals[-1] * np.ones(len(quit_tml)), index=quit_tml)
    tml_portvals = pd.concat([tml_portvals, start_tml, stop_tml]).sort_index()
    # Now plot it all
    test_ax = benchmark.plot(color='black', figsize=figure_size)
    tmanual_portvals.plot(ax=test_ax, color='#00CDFF')
    tml_portvals.plot(ax=test_ax, color='#0DFFBC')
    test_ax.legend(['Benchmark', 'Manual Strategy', 'Machine Learning Strategy'])
    test_ax.set_title('All Strategies- Out of Sample',
                    fontname='Helvetica Neue LT Std', fontsize=plot_title_size)
    test_ax.set_ylabel('Normalized Portfolio Value', fontname='Helvetica Neue LT Std', fontsize=plot_axis_label_size)
    # Print portfolio stats
    # Get statistics for the two strategies
    bench_cum_ret, bench_avg_daily_ret, bench_std_daily_ret, bench_sharpe_ratio = assess_portfolio(benchmark)
    man_ret, man_avg_daily_ret, man_std_daily_ret, man_sharpe_ratio = assess_portfolio(tmanual_portvals.ix[:,0])
    ml_ret, ml_avg_daily_ret, ml_std_daily_ret, ml_sharpe_ratio = assess_portfolio(tml_portvals.ix[:,0])
    # Print the statistics of the two portfolios
    print "Cumulative Return of Benchmark: {}".format(bench_cum_ret)
    print "Cumulative Return of Manual Strategy: {}".format(man_ret)
    print "Cumulative Return of Machine Learning Strategy: {}".format(ml_ret)
    print
    print "Standard Deviation of Benchmark: {}".format(bench_std_daily_ret)
    print "Standard Deviation of Manual Strategy: {}".format(man_std_daily_ret)
    print "Standard Deviation of Machine Learning Strategy: {}".format(ml_std_daily_ret)
    print
    print "Average Daily Return of Benchmark: {}".format(bench_avg_daily_ret)
    print "Average Daily Return of Manual Strategy: {}".format(man_avg_daily_ret)
    print "Average Daily Return of Machine Learning Strategy: {}".format(ml_avg_daily_ret)
    if save_figs:
        test_fig = plt.gcf()
        test_fig.savefig("{}_Test_Strategies.png".format(str(symbol[0])), dpi=100, bbox_inches='tight')

if show_figs:
    plt.show()
