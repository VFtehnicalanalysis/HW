import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from datetime import datetime

import csv_downloader as dwn
#import market_coefficients.py as mcoef
#import market_plots.py as mplt

def main():
    # Список тикеров
    tickers = ['LKOH', 'SBER']
    indexes = ['MOEXREPO', 'IMOEX']
    
    # Даты начала и конца периода
    start_date = datetime(2014, 5, 9)
    end_date = datetime(2024, 5, 9)

    # Загрузка и сохранение данных
    #dwn.download_and_save_data(tickers, start_date, end_date)
    #dwn.download_index_data(indexes, start_date, end_date)

    # Чтение данных и проверка наличия столбца 'close'
    dfs = {}
    for ticker in tickers:
        file_name = f"data/{ticker}_data.csv"
        df = pd.read_csv(file_name, parse_dates=['begin'], index_col='begin')
        if 'close' not in df.columns:
            raise KeyError(f"Column 'close' not found in {file_name}")
        dfs[ticker] = df

    repo_file = "data/MOEXREPO_data.csv"
    imoex_file = "data/IMOEX_data.csv"

    df_repo = pd.read_csv(repo_file, parse_dates=['begin'], index_col='begin')
    df_imoex = pd.read_csv(imoex_file, parse_dates=['begin'], index_col='begin')

    # Рассчитываем лог. доходность
    df_imoex['log_return'] = np.log(df_imoex['close'] / df_imoex['close'].shift(1))
    market_return = df_imoex['log_return'].mean() * 252

    # Используем поле 'close' MOEXREPO как безрисковую ставку
    risk_free_rate = df_repo['close'].mean()

    # Объединение данных
    data = pd.concat([dfs[ticker]['log_return'] for ticker in tickers], axis=1)
    data.columns = tickers
    data.dropna(inplace=True)

    # Параметры
    window = 252
    output_dir = "market_result"

    # Построим графики для IMOEX
    plot_cal_and_ef(data, risk_free_rate, output_file=os.path.join(output_dir, "cal_ef_imo.png"))
    plot_cml(data, risk_free_rate, market_return, output_file=os.path.join(output_dir, "cml_imo.png"))

    betas_imo, expected_returns_imoex = calculate_beta_and_expected_returns(data, market_return, risk_free_rate)
    plot_sml(betas_imo, expected_returns_imoex, risk_free_rate, market_return, output_file=os.path.join(output_dir, "sml_imo.png"))

    treynor_ratios_imoes = calculate_treynor_ratio(data, risk_free_rate)
    plot_treynor_ratio(treynor_ratios_imoex, output_file=os.path.join(output_dir, "treynor_ratio_imoex.png"))

    sortino_ratios_imoex = calculate_sortino_ratio(data, risk_free_rate)
    plot_sortino_ratio(sortino_ratios_imoex, output_file=os.path.join(output_dir, "sortino_ratio_imoex.png"))

    jensen_alphas_imoex = calculate_jensen_alpha(data, market_return, risk_free_rate)
    plot_jensen_alpha(jensen_alphas_imoex, output_file=os.path.join(output_dir, "jensen_alpha_imoex.png"))

    plot_dynamic_treynor_ratio(data, dfs, tickers, window, risk_free_rate, output_file=os.path.join(output_dir, "treynor_ratio_dynamic_imo.png"))
    plot_dynamic_sortino_ratio(data, dfs, tickers, window, risk_free_rate, output_file=os.path.join(output_dir, "sortino_ratio_dynamic_imo.png"))
    plot_dynamic_jensen_alpha(data, dfs, tickers, window, risk_free_rate, market_return, output_file=os.path.join(output_dir, "jensen_alpha_dynamic_imo.png"))

def get_statistics(weights, returns):
    weights = np.array(weights)  # Преобразование weights в массив NumPy
    portfolio_return = np.dot(weights, returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return np.array([portfolio_return, portfolio_volatility])

def plot_cal_and_ef(returns, risk_free_rate=0.03, output_file="cal_ef.png"):
    num_assets = len(returns.columns)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # Найти оптимальные веса
    result = minimize(lambda weights: get_statistics(weights, returns)[1], num_assets*[1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)

    target_returns = np.linspace(returns.mean().min(), returns.mean().max(), 100)
    target_volatilities = []
    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: get_statistics(x, returns)[0] - target_return},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        result = minimize(lambda weights: get_statistics(weights, returns)[1], num_assets*[1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        target_volatilities.append(result.fun)

    plt.figure()
    plt.plot(target_volatilities, target_returns, linestyle='-', color='black', label='Efficient Frontier')

    weights = result.x
    opt_return, opt_volatility = get_statistics(weights, returns)
    max_sharpe_ratio = (opt_return - risk_free_rate) / opt_volatility
    x = np.linspace(0, max(target_volatilities), 100)
    plt.plot(x, risk_free_rate + max_sharpe_ratio * x, linestyle='--', color='blue', label='Capital Allocation Line (CAL)')

    for ticker in returns.columns:
        plt.scatter(np.sqrt(np.var(returns[ticker]) * 252), np.mean(returns[ticker]) * 252, label=ticker)

    plt.xlabel('Portfolio Standard Deviation (Risk)')
    plt.ylabel('Portfolio Expected Return')
    plt.legend()
    plt.grid(True)
    plt.title('Efficient Frontier and CAL')
    plt.savefig(output_file)
    plt.close()

def plot_cml(returns, risk_free_rate=0.03, market_return=0.1, output_file="cml.png"):
    plt.figure()
    portfolio_stats = get_statistics(np.array([0.5, 0.5]), returns)
    market_volatility = portfolio_stats[1]

    x = np.linspace(0, 2 * market_volatility, 100)
    y = risk_free_rate + (market_return - risk_free_rate) * (x / market_volatility)
    plt.plot(x, y, linestyle='--', color='green', label='Capital Market Line (CML)')

    plt.xlabel('Portfolio Standard Deviation (Risk)')
    plt.ylabel('Portfolio Expected Return')
    plt.legend()
    plt.grid(True)
    plt.title('Capital Market Line (CML)')
    plt.savefig(output_file)
    plt.close()

def calculate_beta_and_expected_returns(returns, market_return, risk_free_rate=0.03):
    betas = {}
    expected_returns = {}
    market_cov = returns.cov().loc[returns.columns[0], returns.columns[1]]
    market_var = returns.cov().loc[returns.columns[1], returns.columns[1]]
    beta = market_cov / market_var
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    for ticker in returns.columns:
        betas[ticker] = beta
        expected_returns[ticker] = expected_return
    return betas, expected_returns

def plot_sml(betas, expected_returns, risk_free_rate=0.03, market_return=0.1, output_file="sml.png"):
    plt.figure()
    plt.scatter(list(betas.values()), list(expected_returns.values()), marker='o', c='blue', label='Assets')
    x = np.linspace(0, 2, 100)
    y = risk_free_rate + (market_return - risk_free_rate) * x
    plt.plot(x, y, linestyle='--', color='red', label='Security Market Line (SML)')

    plt.xlabel('Beta')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.grid(True)
    plt.title('Security Market Line (SML)')
    plt.savefig(output_file)
    plt.close()

def plot_dynamic_indicator(ticker, ticker_data, indicator, indicator_label, output_file):
    if len(ticker_data.index) > len(indicator):
        ticker_data = ticker_data.iloc[-len(indicator):]

    mean_indicator = indicator.mean()

    fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax[0].plot(ticker_data.index, ticker_data['close'], label=f'{ticker} Price', color='blue')
    ax[0].set_ylabel('Price')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(ticker_data.index, indicator, label=indicator_label, color='red')
    ax[1].axhline(y=mean_indicator, color='green', linestyle='--', label=f'Mean {indicator_label}')
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel(indicator_label)
    ax[1].legend()
    ax[1].grid(True)

    plt.suptitle(f'{ticker} Price and {indicator_label}')
    plt.savefig(output_file)
    plt.close()
    
def calculate_treynor_ratio(returns, risk_free_rate=0.03):
    betas, expected_returns = calculate_beta_and_expected_returns(returns, risk_free_rate)
    portfolio_return = returns.mean().mean() * 252
    treynor_ratios = {ticker: (expected_returns[ticker] - risk_free_rate) / betas[ticker] for ticker in returns.columns}
    return treynor_ratios

def calculate_sortino_ratio(returns, risk_free_rate=0.03):
    negative_returns = returns[returns < 0]
    downside_std = np.sqrt((negative_returns ** 2).mean()) * np.sqrt(252)
    portfolio_return = returns.mean().mean() * 252
    sortino_ratios = {ticker: (returns[ticker].mean() * 252 - risk_free_rate) / downside_std[ticker] for ticker in returns.columns}
    return sortino_ratios

def calculate_jensen_alpha(returns, market_return, risk_free_rate=0.03):
    betas, expected_returns = calculate_beta_and_expected_returns(returns, market_return, risk_free_rate)
    jensen_alphas = {ticker: returns[ticker].mean() * 252 - (risk_free_rate + betas[ticker] * (market_return - risk_free_rate)) for ticker in returns.columns}
    return jensen_alphas


def plot_dynamic_treynor_ratio(data, dfs, tickers, window, risk_free_rate=0.03, output_file="treynor_ratio_dynamic.png"):
    for ticker in tickers:
        log_returns = data[ticker]
        market_returns = data['IMOEX']
        treynor_ratios = calculate_dynamic_treynor_ratio(log_returns, market_returns, window, risk_free_rate)
        plot_dynamic_indicator(ticker, dfs[ticker], treynor_ratios, 'Treynor Ratio', output_file)

def plot_dynamic_sortino_ratio(data, dfs, tickers, window, risk_free_rate=0.03, output_file="sortino_ratio_dynamic.png"):
    for ticker in tickers:
        log_returns = data[ticker]
        sortino_ratios = calculate_dynamic_sortino_ratio(log_returns, window, risk_free_rate)
        plot_dynamic_indicator(ticker, dfs[ticker], sortino_ratios, 'Sortino Ratio', output_file)

def plot_dynamic_jensen_alpha(data, dfs, tickers, window, risk_free_rate=0.03, market_return=0.1, output_file="jensen_alpha_dynamic.png"):
    for ticker in tickers:
        log_returns = data[ticker]
        market_returns = data['IMOEX']
        jensen_alphas = calculate_dynamic_jensen_alpha(log_returns, market_returns, window, risk_free_rate, market_return)
        plot_dynamic_indicator(ticker, dfs[ticker], jensen_alphas, 'Jensen Alpha', output_file)

if __name__ == "__main__":
    main()
