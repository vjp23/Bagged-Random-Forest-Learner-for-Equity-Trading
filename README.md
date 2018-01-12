# Bagged Random Forest Learner for Equity Trading
by Vince Petaccio

## Summary

In this project, a bagged random forest learner was implemented from scratch with the goal of outperforming both a simple buy-and-hold trading strategy and a manually implemented technical trading strategy on out-of-sample daily closing price data. Three technical indicators were selected: simple moving average (SMA), Bollinger Bands, and Aroon oscillator. 

The bagged learner was successful in outperforming the benchmark strategy as well as a manually implemented strategy based upon a simple set of rules.

See [Bagged Learner Analysis](https://github.com/vjp23/BaggedForestLearner/blob/master/Bagged%20Learner%20Analysis.pdf "Analysis") for a complete quantitative analysis of the results.

## Running the Code
To run code, set the switches in run_strategies.py to True for each Part to be run, and then run run_strategies.py. For example, set run_indicators to True to run code for Part 1: Technical Indicators.

To produce plots, set show_figs to True. To save plots in the same directory in which the code resides, set save_figs to True.

Ensure that the directory containing this code is in the same location as the data directory.
