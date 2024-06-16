import csv_downloader as dwn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from arch import arch_model
from math import ceil
from scipy.optimize import minimize 

from datetime import datetime

# Список тикеров
tickers = ['LKOH', 'SBER']

# Даты начала и конца периода
start_date = datetime(2004, 1, 1)
end_date = datetime(2023, 12, 31)

dwn.download_and_save_data(tickers, start_date, end_date)

# Объединение данных
data = pd.concat([df1['log_return'], df2['log_return']], axis=1)
data.columns = tickers
data.dropna(inplace=True)

# Функция для расчета доходности и ковариации
def get_statistics(weights, returns):
    portfolio_return = np.dot(weights, returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return np.array([portfolio_return, portfolio_volatility])

# Построение Efficient Frontier и CAL
def plot_cal_and_ef(returns, risk_free_rate=0.03, output_file="cal_ef.png"):
    num_assets = len(returns.columns)
    args = (returns,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(lambda weights: get_statistics(weights, returns)[1], num_assets*[1./num_assets,], method='SLSQP', bounds=bounds, constraints=constraints)

    target_returns = np.linspace(returns.mean().min(), returns.mean().max(), 100)
    target_volatilities = []
    for target_return in target_returns:
        constraints = ({'type': 'eq', 'fun': lambda x: get_statistics(x, returns)[0] - target_return},
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        result = minimize(lambda weights: get_statistics(weights, returns)[1], num_assets*[1./num_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
        target_volatilities.append(result.fun)

    plt.figure()
    # Efficient Frontier
    plt.plot(target_volatilities, target_returns, linestyle='-', color='black', label='Efficient Frontier')

    # Capital Allocation Line (CAL)
    weights = result.x
    opt_return, opt_volatility = get_statistics(weights, returns)
    max_sharpe_ratio = (opt_return - risk_free_rate) / opt_volatility
    x = np.linspace(0, max(target_volatilities), 100)
    plt.plot(x, risk_free_rate + max_sharpe_ratio * x, linestyle='--', color='blue', label='Capital Allocation Line (CAL)')

    # Scatter plot for individual assets
    for ticker in returns.columns:
        plt.scatter(np.sqrt(np.var(returns[ticker]) * 252), np.mean(returns[ticker]) * 252, label=ticker)

    plt.xlabel('Portfolio Standard Deviation (Risk)')
    plt.ylabel('Portfolio Expected Return')
    plt.legend()
    plt.grid(True)
    plt.title('Efficient Frontier and CAL')
    plt.savefig(output_file)
    plt.close()

# Построение SML
def plot_sml(beta, expected_returns, risk_free_rate=0.03, market_return=0.1, output_file="sml.png"):
    plt.figure()
    plt.scatter(beta, expected_returns, marker='o', c='blue', label='Assets')
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

# Построение CML
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

# Рассчитаем бету и доходности для SML
def calculate_beta_and_expected_returns(returns, market_return):
    cov_matrix = returns.cov()
    beta = cov_matrix.loc[tickers[0], tickers[1]] / cov_matrix.loc[tickers[1], tickers[1]]
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    return beta, expected_return

# Параметры
risk_free_rate = 0.03
market_return = 0.1

# Построим графики
plot_cal_and_ef(data, risk_free_rate, output_file="cal_ef.png")
plot_cml(data, risk_free_rate, market_return, output_file="cml.png")

# Рассчитаем бету и ожидаемую доходность для каждого актива
beta = {}
expected_returns = {}
for ticker in tickers:
    beta[ticker], expected_returns[ticker] = calculate_beta_and_expected_returns(data, market_return)

# Построим SML
plot_sml(list(beta.values()), list(expected_returns.values()), risk_free_rate, market_return, output_file="sml.png")

# Чтение данных и проверка наличия столбца 'close'
dfs = {}
for ticker in tickers:
    file_name = f"{ticker}_data.csv"
    df = pd.read_csv(file_name, parse_dates=['begin'], index_col='begin')
    if 'close' not in df.columns:
        raise KeyError(f"Column 'close' not found in {file_name}")
    dfs[ticker] = df

# Объединение данных
data = pd.concat([dfs[ticker]['log_return'] for ticker in tickers], axis=1)
data.columns = tickers
data.dropna(inplace=True)

# Функция для расчета доходности и ковариации
def get_statistics(weights, returns):
    portfolio_return = np.dot(weights, returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return np.array([portfolio_return, portfolio_volatility])

# Функция для динамического построения графика цены и показателей
def plot_dynamic_indicator(ticker, ticker_data, indicator, indicator_label, output_file):
    # Проверка длины индикатора и данных
    if len(ticker_data.index) > len(indicator):
        ticker_data = ticker_data.iloc[-len(indicator):]

    # Рассчитаем средний показатель
    mean_indicator = indicator.mean()

    fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # График цены акции
    ax[0].plot(ticker_data.index, ticker_data['close'], color='blue')
    ax[0].set_title(f'Price of {ticker}')
    ax[0].set_ylabel('Price')
    
    # График динамического показателя
    ax[1].plot(ticker_data.index, indicator, color='red', label=f'Dynamic {indicator_label}')
    ax[1].axhline(y=mean_indicator, color='green', linestyle='--', label=f'Average {indicator_label}')
    ax[1].set_title(f'Dynamic {indicator_label}')
    ax[1].set_ylabel(indicator_label)
    ax[1].legend()
    
    plt.savefig(output_file)
    plt.close()

# Построение динамических графиков для CAL, SML и CML
def plot_dynamic_cal(data, window=252, risk_free_rate=0.03, output_file="cal_dynamic.png"):
    rolling_sharpe_ratios = []
    for end in range(window, len(data)):
        rolling_data = data.iloc[end - window:end]
        num_assets = len(rolling_data.columns)
        bounds = tuple((0, 1) for asset in range(num_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        result = minimize(lambda weights: get_statistics(weights, rolling_data)[1], num_assets*[1./num_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x
        portfolio_return, portfolio_volatility = get_statistics(weights, rolling_data)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        rolling_sharpe_ratios.append(sharpe_ratio)

    # Приведение индексов к одному размеру
    rolling_sharpe_ratios = pd.Series(rolling_sharpe_ratios, index=data.index[window:])
    
    plot_dynamic_indicator(tickers[0], dfs[tickers[0]], rolling_sharpe_ratios, 'Sharpe Ratio', output_file)

def plot_dynamic_sml(ticker_data, market_data, window=252, risk_free_rate=0.03, market_return=0.1, output_file="sml_dynamic.png"):
    rolling_betas = []
    for end in range(window, len(ticker_data)):
        rolling_data = ticker_data.iloc[end - window:end]
        rolling_market_data = market_data.iloc[end - window:end]

        # Убедимся, что нет пропусков в данных
        rolling_data['log_return'] = rolling_data['log_return'].ffill()
        rolling_market_data['log_return'] = rolling_market_data['log_return'].ffill()

        # Проверка наличия пропусков
        if len(rolling_data) != len(rolling_market_data):
            continue
        
        # Рассчитываем ковариационную матрицу
        cov_matrix = np.cov(rolling_data['log_return'], rolling_market_data['log_return'])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        rolling_betas.append(beta)
    
    # Приведение индексов к одному размеру
    rolling_betas = pd.Series(rolling_betas, index=ticker_data.index[window:window+len(rolling_betas)])
    
    # Построение графика
    plot_dynamic_indicator(tickers[0], ticker_data, rolling_betas, 'Beta', output_file)


def plot_dynamic_cml(data_files, market_data_file, window=252, risk_free_rate=0.03, market_return=0.1, output_file="cml_dynamic.png"):
    data = [pd.read_csv(file, index_col='begin', parse_dates=True) for file in data_files]
    market_data = pd.read_csv(market_data_file, index_col='begin', parse_dates=True)

    combined_data = pd.concat([df['log_return'] for df in data], axis=1)
    combined_data.columns = [os.path.splitext(os.path.basename(file))[0] for file in data_files]

    rolling_returns = []
    rolling_volatilities = []

    for end in range(window, len(combined_data)):
        rolling_data = combined_data.iloc[end - window:end].copy()
        rolling_market_data = market_data.iloc[end - window:end].copy()

        rolling_data = rolling_data.ffill()
        rolling_market_data = rolling_market_data.ffill()

        num_assets = rolling_data.shape[1]
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    # Минимизация волатильности портфеля
        result = minimize(lambda weights: get_statistics(weights, rolling_data)[1], 
                          num_assets * [1. / num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x
        portfolio_return, portfolio_volatility = get_statistics(weights, rolling_data)
        rolling_returns.append(portfolio_return)
        rolling_volatilities.append(portfolio_volatility)

    # Приведение индексов к одному размеру
    rolling_returns = pd.Series(rolling_returns, index=combined_data.index[window:])
    rolling_volatilities = pd.Series(rolling_volatilities, index=combined_data.index[window:])

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_volatilities, rolling_returns, label='Dynamic Portfolio')
    
    # Построение CML
    market_std = market_data['log_return'].std() * np.sqrt(252)
    cml_x = [0, max(rolling_volatilities)]
    cml_y = [risk_free_rate, risk_free_rate + max(cml_x) * (market_return - risk_free_rate) / market_std]
    plt.plot(cml_x, cml_y, label='CML', color='red')

    plt.title('Dynamic Capital Market Line (CML)')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.legend()
    plt.savefig(output_file)
    plt.show()

# Параметры
risk_free_rate = 0.03
market_return = 0.1
window_size = 252  # Например, 252 дня (один год)

dwn.download_index_data('IMOEX', start_date, end_date)
dwn.download_index_data('RGBI', start_date, end_date)
market_data = pd.read_csv("IMOEX_data.csv", parse_dates=['begin'], index_col='begin')  # предполагаем, что у вас есть данные по рыночному индексу
# Данные рынка (используем средний индекс как рынок)
market_data = data.mean(axis=1).to_frame(name='log_return')

# Заполним пропущенные значения последним известным
market_data['log_return'] = market_data['log_return'].ffill()

# Построим динамические графики
plot_dynamic_cal(data, window=window_size, risk_free_rate=risk_free_rate, output_file="cal_dynamic.png")
plot_dynamic_sml(dfs[tickers[0]], market_data, window=window_size, risk_free_rate=risk_free_rate, market_return=market_return, output_file="sml_dynamic.png")
data_files = [f'{ticker}_data.csv' for ticker in tickers]
index = 'IMOEX'
market_data_file = f'{index}_data.csv'

plot_dynamic_cml(data_files, market_data_file)

def main():
    #VAR
    #VAR (T-student)
    #VAR (EWMA)
    #VAR ()
    #Expected Shortfall (синоним Expected Tail Loss).
    #print(get_descriptive_stats(df))
    #base_plots(df)
    #drowdown(df)
    #EWMA(df)

    #log_returns = df['log_return'].dropna()
    #var = calculate_parametric_var(log_returns)
    #plot_log_returns_with_var_signals(df, var)
    #plot_interactive_price_with_var_signals(df, var)

    #parametric_var = calculate_parametric_var(log_returns)
    #ewma_var = calculate_ewma_var(log_returns)
    #hull_white_var = calculate_hull_white_var(log_returns)
    #plot_interactive_price_with_var_signals(df, parametric_var, ewma_var, hull_white_var)

    # Расчет VaR и ES
    #VaR, ES, results = backtest_var_es(df)
    
    # Вывод результатов бэктеста
    #results_df = pd.DataFrame(results, columns=["Method", "Violation Ratio", "Volatility"])
    #print(results_df)

    # Построение графика бэктеста
    #plot_backtest(df, VaR, 1000)
    
    # Построение графика дополнительных тестов
    #plot_additional_tests(df, VaR, 1000)

    #display_results(results)
    print(1)
    


if __name__ == "__main__":
    main()