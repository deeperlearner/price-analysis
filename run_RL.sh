#!/bin/bash
# Python version: 3.6.X
# Needed packages: matplotlib, mpl_finance, numpy, pandas, requests, scikit-learn, seaborn, talib, tensorflow==1.13.1
python3 get_data.py --crypto BTC --tick 1h --time_range week
# python3 data_preprocessing.py --crypto BTC --tick 1h --time_range week
