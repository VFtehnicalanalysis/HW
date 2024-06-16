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

#dwn.download_and_save_data(tickers, start_date, end_date)

WT = 0 #Testing window size
WE = 0 #Estimation window size 
T=WE+WT #Number of observations in a sample 
nu = 0 #Indicates whether a violation occurs
v = 0 #Count of violations

df1 = pd.read_csv("LKOH_data.csv", parse_dates=['begin'], index_col='begin')
df2 = pd.read_csv("SBER_data.csv", parse_dates=['begin'], index_col='begin')
for df in [df1, df2]:
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

def get_descriptive_stats(df):
    #if 'log_return' not in df.columns:
    #    df.loc[:, 'log_return'] = np.log(df['close'] / df['close'].shift(1))
    stats = {
          'mean': df["log_return"].mean(),
          'median': df["log_return"].median(),
          'skew': df["log_return"].skew(),
          'kurtosis': df["log_return"].kurt(),
          'volatility': df["log_return"].std(),
          'value_at_risk_95': df["log_return"].quantile(0.05),
          'expected shortfall':df[df["log_return"]<df["log_return"].quantile(0.05)]["log_return"].mean()
      }

    return stats

def base_plots(df):
    plt.hist(df["log_return"],bins=50)
    plt.savefig('histogram.png')

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)
    df["log_return"].plot(ax=ax1, label='Log Return')
    (df["log_return"].rolling(window=50).mean().dropna()).plot(ax=ax1, color="red", label='50-day MA')
    ax1.set_ylabel('Log Return & Volume MA')
    ax1.legend()

    (df["log_return"].rolling(window=30).std().dropna()).plot(ax=ax2, label='30-day Vol MA')
    ax2.set_ylabel('Volume MA')
    ax2.legend()
    plt.savefig('log_return.png')

def drowdown(df):
    PORTF=100*(1+df["log_return"]).cumprod()
    PORTF.tail()
    #PEAKS variable stores max value of PORTF. When PORTF grows, PEAKS updates.
    PEAKS=PORTF.cummax()
    PEAKS.head(20)
    DRAWDOWN=(PORTF-PEAKS)/PEAKS
    DRAWDOWN.plot(kind="line")
    DRAWDOWN["2022":].min()
    DRAWDOWN["2022":].idxmin()

    # Создаем новый объект Figure и две панели Axes
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)

    # Первая панель: график DRAWDOWN
    DRAWDOWN.plot(ax=ax1, label='Drawdown')
    ax1.plot(DRAWDOWN.idxmin(), DRAWDOWN["2022":].min(), 'ro', label='Max Drawdown')  # добавляем точку минимума
    ax1.annotate(f'Min Drawdown: {DRAWDOWN["2022":].min():.2%}', 
                xy=(DRAWDOWN.idxmin(), DRAWDOWN["2022":].min()), 
                xytext=(DRAWDOWN.idxmin(), DRAWDOWN["2022":].min() + 0.02), 
                arrowprops=dict(facecolor='black', arrowstyle='->'),  # добавляем аннотацию к точке минимума
                )
    ax1.set_ylabel('Drawdown')
    ax1.legend()

    # Вторая панель: графики PORTF и PEAKS
    PORTF.plot(ax=ax2, label='PORTF')
    PEAKS.plot(ax=ax2, style='g--', label='Peaks')  # отметка Peaks
    ax2.set_ylabel('PORTF')
    ax2.legend()

    # Сохраняем график в файл
    plt.savefig('drawdown_and_portf.png')

def EWMA(df):
    # Создаем новый объект Figure и Axes
    fig, ax = plt.subplots(figsize=(15, 6))

    # Построение графика сглаженной экспоненциальной скользящей волатильности
    stock_ewm_vol = df["log_return"].ewm(alpha=0.95).std()
    stock_ewm_vol.plot(ax=ax, label='EWM Volatility')  # добавляем график на тот же ax

    # Добавление легенды
    ax.legend()

    # Сохраняем график в файл
    plt.savefig('EWM.png')

def plot_log_returns_with_var_signals(df, var):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['log_return'], label='Log Return', color='blue')
    
    var_signals = df[df['log_return'] < var]
    plt.scatter(var_signals.index, var_signals['log_return'], color='red', label='VAR Signal', marker='o')
    
    plt.title('Log Returns with Parametric VAR Signals')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.legend()
    #plt.show()
    plt.savefig('parametric_var.png')

def calculate_parametric_var(log_returns, confidence_level=0.95, degrees_freedom=10):
    mean = np.mean(log_returns)
    std_dev = np.std(log_returns)
    t_value = stats.t.ppf(1 - confidence_level, degrees_freedom)
    var = mean + t_value * std_dev
    return var
def calculate_ewma_var(log_returns, confidence_level=0.95, lambda_factor=0.94):

    # Вычисляем EWMA дисперсию
    ewma_variance = log_returns.ewm(span=(1 / (1 - lambda_factor))).var()
    ewma_std_dev = np.sqrt(ewma_variance)
    
    # Рассчитываем VaR
    z_score = stats.norm.ppf(1 - confidence_level)
    ewma_var = z_score * ewma_std_dev
    
    return ewma_var
def calculate_hull_white_var(log_returns, confidence_level=0.95, lambda_factor=0.94):
    # Среднее значение логарифмических доходностей
    mean = log_returns.mean()
    
    # Вычисляем EWMA дисперсию
    ewma_variance = log_returns.ewm(span=(1 / (1 - lambda_factor))).var()
    ewma_std_dev = np.sqrt(ewma_variance)
    
    # Рассчитываем VaR
    z_score = stats.norm.ppf(1 - confidence_level)
    hull_white_var = mean + z_score * ewma_std_dev
    
    return hull_white_var
def plot_interactive_price_with_var_signals(df, parametric_var, ewma_var, hull_white_var):
    # Создаем фигуру
    fig = go.Figure()

    # Добавляем линию цены закрытия
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        mode='lines',
        name='Closing Price'
    ))

    # Выбираем сигналы Parametric VAR
    parametric_var_signals = df[df['log_return'] < parametric_var]
    fig.add_trace(go.Scatter(
        x=parametric_var_signals.index,
        y=parametric_var_signals['close'],
        mode='markers',
        name='Parametric VAR Signal',
        marker=dict(color='red', size=8, symbol='circle')
    ))

    # Выбираем сигналы EWMA VAR
    ewma_var_signals = df[df['log_return'] < ewma_var]
    fig.add_trace(go.Scatter(
        x=ewma_var_signals.index,
        y=ewma_var_signals['close'],
        mode='markers',
        name='EWMA VAR Signal',
        marker=dict(color='orange', size=8, symbol='circle')
    ))

    # Выбираем сигналы Hull-White VAR
    hull_white_var_signals = df[df['log_return'] < hull_white_var]
    fig.add_trace(go.Scatter(
        x=hull_white_var_signals.index,
        y=hull_white_var_signals['close'],
        mode='markers',
        name='Hull-White VAR Signal',
        marker=dict(color='green', size=8, symbol='circle')
    ))

    # Настройки оформления графика
    fig.update_layout(
        title='Price with VAR Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        hovermode='x unified'
    )

    # Добавляем интерактивность
    fig.update_xaxes(rangeslider_visible=True)
    
    fig.show()

# Бэктест
def backtest_var_es(df, WE=1000, p=0.01, lmbda=0.94):
    y = df['log_return'].values
    T = len(y)
    l1 = ceil(WE * p)
    value = 1
    VaR = np.full([T, 5], np.nan)  # matrix for forecasts (добавлен t-GARCH)
    ES = np.full([T, 2], np.nan)   # matrix for ES forecasts

    # Initial variance for EWMA
    s11 = np.var(y[:WE])
    for t in range(1, WE):
        s11 = lmbda * s11 + (1 - lmbda) * y[t - 1]**2

    for t in range(WE, T):
        t1 = t - WE
        t2 = t - 1
        window = y[t1:t2 + 1]
        s11 = lmbda * s11 + (1 - lmbda) * y[t - 1]**2
        
        # EWMA
        VaR[t, 0] = -stats.norm.ppf(p) * np.sqrt(s11) * value
        ES[t, 0] = np.sqrt(s11) * stats.norm.pdf(stats.norm.ppf(p)) / p
        
        # MA
        VaR[t, 1] = -np.std(window, ddof=1) * stats.norm.ppf(p) * value
        
        # HS
        ys = np.sort(window)
        VaR[t, 2] = -ys[l1 - 1] * value
        ES[t, 1] = -np.mean(ys[:l1]) * value
        
        # GARCH(1,1)
        am_garch = arch_model(window, mean='Zero', vol='Garch', p=1, o=0, q=1, dist='Normal', rescale=False)
        res_garch = am_garch.fit(update_freq=0, disp='off', show_warning=False)
        s4 = res_garch.params['omega'] + res_garch.params['alpha[1]'] * window[-1]**2 + res_garch.params['beta[1]'] * res_garch.conditional_volatility[-1]**2
        VaR[t, 3] = -np.sqrt(s4) * stats.norm.ppf(p) * value

        # t-GARCH
        am_tgarch = arch_model(window, mean='Zero', vol='Garch', p=1, o=0, q=1, dist='t', rescale=False)
        res_tgarch = am_tgarch.fit(update_freq=0, disp='off', show_warning=False)
        s5 = res_tgarch.params['omega'] + res_tgarch.params['alpha[1]'] * window[-1]**2 + res_tgarch.params['beta[1]'] * res_tgarch.conditional_volatility[-1]**2
        nu = res_tgarch.params['nu']
        VaR[t, 4] = -np.sqrt(s5) * stats.t.ppf(p, df=nu) * value

    # Backtesting analysis
    W1 = WE
    results = []
    methods = ["EWMA", "MA", "HS", "GARCH", "t-GARCH"]

    for i in range(5):
        num_violations = sum(y[W1:T] < -VaR[W1:T, i])
        violation_ratio = num_violations / (p * (T - WE))
        avg_volatility = np.std(VaR[W1:T, i], ddof=1)
        results.append([methods[i], num_violations, violation_ratio, avg_volatility])

    return VaR, ES, results

def display_results(results):
    df_results = pd.DataFrame(results, columns=['Method', 'Num Violations', 'Violation Ratio', 'Avg Volatility'])
    print(df_results)

def plot_backtest(df, VaR, WE):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[WE:], df['log_return'][WE:], label='Log Returns', color='black')
    plt.plot(df.index[WE:], -VaR[WE:, 0], label='EWMA VaR', linestyle='--', color='orange')
    plt.plot(df.index[WE:], -VaR[WE:, 1], label='MA VaR', linestyle='--', color='blue')
    plt.plot(df.index[WE:], -VaR[WE:, 2], label='HS VaR', linestyle='--', color='red')
    plt.plot(df.index[WE:], -VaR[WE:, 3], label='GARCH VaR', linestyle='--', color='green')
    plt.plot(df.index[WE:], -VaR[WE:, 4], label='t-GARCH VaR', linestyle='--', color='purple')
    plt.title('Backtesting VaR')
    plt.xlabel('Date')
    plt.ylabel('Returns / VaR')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('backtest1.png')


# Тест покрытия Бернулли
def bernoulli_coverage_test(p, v):
    lv = len(v)
    sv = sum(v)
    al = np.log(p) * sv + np.log(1 - p) * (lv - sv)
    bl = np.log(sv / lv) * sv + np.log(1 - sv / lv) * (lv - sv)
    return -2 * (al - bl)

# Тест независимости
def independence_test(V):
    J = np.full([len(V), 4], 0)
    for i in range(1, len(V) - 1):
        J[i, 0] = (V[i - 1] == 0) & (V[i] == 0)
        J[i, 1] = (V[i - 1] == 0) & (V[i] == 1)
        J[i, 2] = (V[i - 1] == 1) & (V[i] == 0)
        J[i, 3] = (V[i - 1] == 1) & (V[i] == 1)
    V_00 = sum(J[:, 0])
    V_01 = sum(J[:, 1])
    V_10 = sum(J[:, 2])
    V_11 = sum(J[:, 3])
    p_00 = V_00 / (V_00 + V_01)
    p_01 = V_01 / (V_00 + V_01)
    p_10 = V_10 / (V_10 + V_11)
    p_11 = V_11 / (V_10 + V_11)
    hat_p = (V_01 + V_11) / (V_00 + V_01 + V_10 + V_11)
    al = np.log(1 - hat_p) * (V_00 + V_10) + np.log(hat_p) * (V_01 + V_11)
    if p_00 <= 0 or p_01 <= 0 or p_10 <= 0 or p_11 <= 0:
        bl = np.nan  # или другое значение, которое обозначает недопустимый результат
    else:
        bl = np.log(p_00) * V_00 + np.log(p_01) * V_01 + np.log(p_10) * V_10 + np.log(p_11) * V_11
    return -2 * (al - bl)

# Визуализация дополнительных тестов
def plot_additional_tests(df, VaR, WE):
    y = df['log_return'].values
    T = len(y)
    p = 0.01
    V = np.zeros((T, 5))

    for i in range(5):
        V[:, i] = y < -VaR[:, i]

    coverage_results = []
    independence_results = []

    for i in range(5):
        bernoulli_test = bernoulli_coverage_test(p, V[WE:, i])
        ind_test = independence_test(V[WE:, i])
        coverage_results.append(bernoulli_test)
        independence_results.append(ind_test)

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    methods = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5']  # Определяем methods

    axs[0].bar(methods, coverage_results, color=['orange', 'blue', 'red', 'green', 'purple'])
    axs[0].set_title('Bernoulli Coverage Test')
    axs[0].set_ylabel('Test Statistic')

    axs[1].bar(methods, independence_results, color=['orange', 'blue', 'red', 'green', 'purple'])
    axs[1].set_title('Independence Test')
    axs[1].set_ylabel('Test Statistic')

    plt.tight_layout()
    plt.savefig('addition_tests.png')
    plt.show()

# Вывод результатов
def print_results(results):
    df_results = pd.DataFrame(results, columns=['Method', 'Violation Ratio', 'Volatility'])
    print(df_results)

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