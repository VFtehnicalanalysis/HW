import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from datetime import datetime

# Удалены неиспользуемые библиотеки
# import scipy.stats as stats
# import plotly.graph_objects as go
# import math.ceil

def main():
    # Список тикеров
    tickers = ['LKOH', 'SBER']

    # Даты начала и конца периода
    start_date = datetime(2004, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Загрузка и сохранение данных
    import csv_downloader as dwn
    dwn.download_and_save_data(tickers, start_date, end_date)
    dwn.download_index_data('RGBI', start_date, end_date)
    dwn.download_index_data('IMOEX', start_date, end_date)

    # Чтение данных и проверка наличия столбца 'close'
    dfs = {}
    for ticker in tickers:
        file_name = f"data/{ticker}_data.csv"
        df = pd.read_csv(file_name, parse_dates=['begin'], index_col='begin')
        if 'close' not in df.columns:
            raise KeyError(f"Column 'close' not found in {file_name}")
        dfs[ticker] = df

    rgb_file = "data/RGBI_data.csv"
    imoe_file = "data/IMOEX_data.csv"

    df_rgb = pd.read_csv(rgb_file, parse_dates=['begin'], index_col='begin')
    df_imo = pd.read_csv(imoe_file, parse_dates=['begin'], index_col='begin')

    # Объединение данных
    data = pd.concat([dfs[ticker]['log_return'] for ticker in tickers], axis=1)
    data.columns = tickers
    data.dropna(inplace=True)

    # Параметры
    market_return = 0.1
    output_dir = "market_result"

    # Построим графики для RGBI
    risk_free_rate_rgb = df_rgb['log_return'].mean() * 252

    plot_cal_and_ef(data, risk_free_rate_rgb, output_file=os.path.join(output_dir, "cal_ef_rgb.png"))
    plot_cml(data, risk_free_rate_rgb, market_return, output_file=os.path.join(output_dir, "cml_rgb.png"))

    betas_rgb, expected_returns_rgb = calculate_beta_and_expected_returns(data, market_return, risk_free_rate_rgb)
    plot_sml(betas_rgb, expected_returns_rgb, risk_free_rate_rgb, market_return, output_file=os.path.join(output_dir, "sml_rgb.png"))

    treynor_ratios_rgb = calculate_treynor_ratio(data, risk_free_rate_rgb)
    plot_treynor_ratio(treynor_ratios_rgb, output_file=os.path.join(output_dir, "treynor_ratio_rgb.png"))

    sortino_ratios_rgb = calculate_sortino_ratio(data, risk_free_rate_rgb)
    plot_sortino_ratio(sortino_ratios_rgb, output_file=os.path.join(output_dir, "sortino_ratio_rgb.png"))

    jensen_alphas_rgb = calculate_jensen_alpha(data, market_return, risk_free_rate_rgb)
    plot_jensen_alpha(jensen_alphas_rgb, output_file=os.path.join(output_dir, "jensen_alpha_rgb.png"))

    plot_dynamic_treynor_ratio(data, dfs, tickers, window=252, risk_free_rate=risk_free_rate_rgb, output_file=os.path.join(output_dir, "treynor_ratio_dynamic_rgb.png"))
    plot_dynamic_sortino_ratio(data, dfs, tickers, window=252, risk_free_rate=risk_free_rate_rgb, output_file=os.path.join(output_dir, "sortino_ratio_dynamic_rgb.png"))
    plot_dynamic_jensen_alpha(data, dfs, tickers, window=252, risk_free_rate=risk_free_rate_rgb, market_return=market_return, output_file=os.path.join(output_dir, "jensen_alpha_dynamic_rgb.png"))

    # Построим графики для IMOEX
    risk_free_rate_imo = df_imo['log_return'].mean() * 252

    plot_cal_and_ef(data, risk_free_rate_imo, output_file=os.path.join(output_dir, "cal_ef_imo.png"))
    plot_cml(data, risk_free_rate_imo, market_return, output_file=os.path.join(output_dir, "cml_imo.png"))

    betas_imo, expected_returns_imo = calculate_beta_and_expected_returns(data, market_return, risk_free_rate_imo)
    plot_sml(betas_imo, expected_returns_imo, risk_free_rate_imo, market_return, output_file=os.path.join(output_dir, "sml_imo.png"))

    treynor_ratios_imo = calculate_treynor_ratio(data, risk_free_rate_imo)
    plot_treynor_ratio(treynor_ratios_imo, output_file=os.path.join(output_dir, "treynor_ratio_imo.png"))

    sortino_ratios_imo = calculate_sortino_ratio(data, risk_free_rate_imo)
    plot_sortino_ratio(sortino_ratios_imo, output_file=os.path.join(output_dir, "sortino_ratio_imo.png"))

    jensen_alphas_imo = calculate_jensen_alpha(data, market_return, risk_free_rate_imo)
    plot_jensen_alpha(jensen_alphas_imo, output_file=os.path.join(output_dir, "jensen_alpha_imo.png"))

    plot_dynamic_treynor_ratio(data, dfs, tickers, window=252, risk_free_rate=risk_free_rate_imo, output_file=os.path.join(output_dir, "treynor_ratio_dynamic_imo.png"))
    plot_dynamic_sortino_ratio(data, dfs, tickers, window=252, risk_free_rate=risk_free_rate_imo, output_file=os.path.join(output_dir, "sortino_ratio_dynamic_imo.png"))
    plot_dynamic_jensen_alpha(data, dfs, tickers, window=252, risk_free_rate=risk_free_rate_imo, market_return=market_return, output_file=os.path.join(output_dir, "jensen_alpha_dynamic_imo.png"))


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
    ax[0].plot(ticker_data.index, ticker_data['close'], color='blue')
    ax[0].set_title(f'Price of {ticker}')
    ax[0].set_ylabel('Price')

    ax[1].plot(ticker_data.index, indicator, color='red', label=f'Dynamic {indicator_label}')
    ax[1].axhline(y=mean_indicator, color='green', linestyle='--', label=f'Average {indicator_label}')
    ax[1].set_title(f'Dynamic {indicator_label}')
    ax[1].set_ylabel(indicator_label)
    ax[1].legend()

    plt.savefig(output_file)
    plt.close()

def plot_dynamic_cal(data, dfs, tickers, window=252, risk_free_rate=0.03, output_file="cal_dynamic.png"):
    rolling_sharpe_ratios = []
    for end in range(window, len(data)):
        rolling_data = data.iloc[end - window:end]
        num_assets = len(rolling_data.columns)
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        result = minimize(lambda weights: get_statistics(weights, rolling_data)[1], num_assets * [1. / num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x
        portfolio_return, portfolio_volatility = get_statistics(weights, rolling_data)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        rolling_sharpe_ratios.append(sharpe_ratio)

    rolling_sharpe_ratios = pd.Series(rolling_sharpe_ratios, index=data.index[window:])
    plot_dynamic_indicator(tickers[0], dfs[tickers[0]], rolling_sharpe_ratios, 'Sharpe Ratio', output_file)

def plot_dynamic_sml(ticker_data, market_data, tickers, window=252, risk_free_rate=0.03, market_return=0.1, output_file="sml_dynamic.png"):
    rolling_betas = []
    for end in range(window, len(ticker_data)):
        rolling_data = ticker_data.iloc[end - window:end]
        rolling_market_data = market_data.iloc[end - window:end]
        cov_matrix = np.cov(rolling_data, rolling_market_data)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        rolling_betas.append(beta)

    rolling_betas = pd.Series(rolling_betas, index=ticker_data.index[window:])
    expected_returns = risk_free_rate + rolling_betas * (market_return - risk_free_rate)
    plot_dynamic_indicator(tickers[0], ticker_data, expected_returns, 'Expected Return', output_file)

def plot_dynamic_cml(data, dfs, tickers, window=252, risk_free_rate=0.03, market_return=0.1, output_file="cml_dynamic.png"):
    rolling_returns = []
    for end in range(window, len(data)):
        rolling_data = data.iloc[end - window:end]
        portfolio_return, portfolio_volatility = get_statistics([0.5, 0.5], rolling_data)
        rolling_returns.append(portfolio_return)

    rolling_returns = pd.Series(rolling_returns, index=data.index[window:])
    plot_dynamic_indicator(tickers[0], dfs[tickers[0]], rolling_returns, 'Expected Return', output_file)

def calculate_treynor_ratio(returns, risk_free_rate=0.03):
    betas, expected_returns = calculate_beta_and_expected_returns(returns, risk_free_rate)
    treynor_ratios = {ticker: (returns[ticker].mean() * 252 - risk_free_rate) / betas[ticker] for ticker in returns.columns}
    return treynor_ratios

def calculate_sortino_ratio(returns, risk_free_rate=0.03):
    negative_returns = returns[returns < 0]
    downside_std = np.sqrt((negative_returns ** 2).mean()) * np.sqrt(252)
    sortino_ratios = {ticker: (returns[ticker].mean() * 252 - risk_free_rate) / downside_std[ticker] for ticker in returns.columns}
    return sortino_ratios

def calculate_jensen_alpha(returns, market_return, risk_free_rate=0.03):
    betas, expected_returns = calculate_beta_and_expected_returns(returns, market_return, risk_free_rate)
    jensen_alphas = {ticker: returns[ticker].mean() * 252 - (risk_free_rate + betas[ticker] * (market_return - risk_free_rate)) for ticker in returns.columns}
    return jensen_alphas

def plot_treynor_ratio(treynor_ratios, output_file="treynor_ratio.png"):
    plt.figure()
    plt.bar(treynor_ratios.keys(), treynor_ratios.values(), color='blue')
    plt.xlabel('Tickers')
    plt.ylabel('Treynor Ratio')
    plt.title('Treynor Ratio for Assets')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def plot_sortino_ratio(sortino_ratios, output_file="sortino_ratio.png"):
    plt.figure()
    plt.bar(sortino_ratios.keys(), sortino_ratios.values(), color='green')
    plt.xlabel('Tickers')
    plt.ylabel('Sortino Ratio')
    plt.title('Sortino Ratio for Assets')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def plot_jensen_alpha(jensen_alphas, output_file="jensen_alpha.png"):
    plt.figure()
    plt.bar(jensen_alphas.keys(), jensen_alphas.values(), color='red')
    plt.xlabel('Tickers')
    plt.ylabel("Jensen's Alpha")
    plt.title("Jensen's Alpha for Assets")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def plot_dynamic_treynor_ratio(data, dfs, tickers, window=252, risk_free_rate=0.03, output_file="treynor_ratio_dynamic.png"):
    rolling_treynor_ratios = []
    for end in range(window, len(data)):
        rolling_data = data.iloc[end - window:end]
        treynor_ratios = calculate_treynor_ratio(rolling_data, risk_free_rate)
        rolling_treynor_ratios.append(treynor_ratios[tickers[0]])
    rolling_treynor_ratios = pd.Series(rolling_treynor_ratios, index=data.index[window:])
    plot_dynamic_indicator(tickers[0], dfs[tickers[0]], rolling_treynor_ratios, 'Treynor Ratio', output_file)

def plot_dynamic_sortino_ratio(data, dfs, tickers, window=252, risk_free_rate=0.03, output_file="sortino_ratio_dynamic.png"):
    rolling_sortino_ratios = []
    for end in range(window, len(data)):
        rolling_data = data.iloc[end - window:end]
        sortino_ratios = calculate_sortino_ratio(rolling_data, risk_free_rate)
        rolling_sortino_ratios.append(sortino_ratios[tickers[0]])
    rolling_sortino_ratios = pd.Series(rolling_sortino_ratios, index=data.index[window:])
    plot_dynamic_indicator(tickers[0], dfs[tickers[0]], rolling_sortino_ratios, 'Sortino Ratio', output_file)

def plot_dynamic_jensen_alpha(data, dfs, tickers, window=252, risk_free_rate=0.03, market_return=0.1, output_file="jensen_alpha_dynamic.png"):
    rolling_jensen_alphas = []
    for end in range(window, len(data)):
        rolling_data = data.iloc[end - window:end]
        jensen_alphas = calculate_jensen_alpha(rolling_data, market_return, risk_free_rate)
        rolling_jensen_alphas.append(jensen_alphas[tickers[0]])
    rolling_jensen_alphas = pd.Series(rolling_jensen_alphas, index=data.index[window:])
    plot_dynamic_indicator(tickers[0], dfs[tickers[0]], rolling_jensen_alphas, "Jensen's Alpha", output_file)





if __name__ == "__main__":
    main()
