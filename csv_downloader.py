import sys
import os
import requests
import apimoex
import pandas as pd
import numpy as np
from datetime import datetime

def download_and_save_data(tickers, start_date, end_date):
    with requests.Session() as session:
        for ticker in tickers:
            request_params = {
                'interval': 24,  # Дневные свечи
                'start': start_date,
                'end': end_date,
                'market': 'shares'
            }
            data = apimoex.get_market_candles(session, security=ticker, **request_params)
            df = pd.DataFrame(data)
            df['begin'] = pd.to_datetime(df['begin'])
            df.set_index('begin', inplace=True)
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            file_name = f'data/{ticker}_data.csv'
            if os.path.exists(file_name):
                os.remove(file_name)
                print(f'File {file_name} already exists. Removed.')
            df.dropna(inplace=True)
            df.to_csv(file_name)
            print(f'Data for {ticker} saved to {file_name}')

def download_index_data(index, start_date, end_date):
    with requests.Session() as session:
        request_params = {
            'interval': 24,  # Дневные свечи
            'start': start_date,
            'end': end_date,
            'market': 'index'
        }
        data = apimoex.get_market_candles(session, security=index, **request_params)
        df = pd.DataFrame(data)
        df['begin'] = pd.to_datetime(df['begin'])
        df.set_index('begin', inplace=True)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        file_name = f'data/{index}_data.csv'
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f'File {file_name} already exists. Removed.')
        df.dropna(inplace=True)
        df.to_csv(file_name)
        print(f'Data for {index} saved to {file_name}')