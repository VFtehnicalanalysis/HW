import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Шаг 1: Загрузка данных и преобразование
file_path = 'Loan_Portfolio.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'], dayfirst=True)
df['AMT_PRINCIPAL'] = df['AMT_PRINCIPAL'].str.replace(' ', '').str.replace(',', '').astype(float)
transformed_file_path = 'Transformed_Loan_Portfolio.csv'
df.to_csv(transformed_file_path, index=False)

# Шаг 2: Загрузка преобразованных данных для дальнейшей обработки
df_transformed = pd.read_csv(transformed_file_path)
df_transformed.sort_values(by=['CONTRACT_ID', 'REPORT_DATE'], inplace=True)

# Шаг 3: Расчет матрицы миграции
def calculate_migration_matrix(df):
    df['NEXT_DPD_GROUP'] = df.groupby('CONTRACT_ID')['DPD_GROUP'].shift(-1)
    migration_matrix = pd.crosstab(df['DPD_GROUP'], df['NEXT_DPD_GROUP'], normalize='index')
    return migration_matrix

migration_matrix = calculate_migration_matrix(df_transformed)

# Проверка наличия дефолтной группы
print("Колонки матрицы миграций:", migration_matrix.columns)
default_states = ['180+', '91 - 120']
# Поиск доступного дефолтного состояния
default_state = next((state for state in default_states if state in migration_matrix.columns), None)
if default_state is None:
    raise ValueError("Дефолтное состояние не найдено в матрице миграций. Проверьте данные.")

# Шаг 5: Расчет PD (12 месяцев и Lifetime)
def calculate_pd(migration_matrix, default_state):
    pd_12m = migration_matrix.loc[:, default_state].sum()
    return pd_12m

pd_12m = calculate_pd(migration_matrix, default_state)

# Расчет PD Lifetime
def calculate_lifetime_pd(pd_12m, period_years):
    pd_lifetime = 1 - (1 - pd_12m) ** period_years
    return pd_lifetime

period_years = 5  # допустим, что срок жизни кредита составляет 5 лет
pd_lifetime = calculate_lifetime_pd(pd_12m, period_years)

# Шаг 6: Расчет ECL
def calculate_ecl(df, pd_value, lgd=0.45, ead_col='AMT_PRINCIPAL'):
    df['ECL'] = df[ead_col] * pd_value * lgd
    return df['ECL'].sum()

ecl = calculate_ecl(df_transformed, pd_lifetime)  # Используем Lifetime PD для расчета ECL

# Шаг 7: Расчет коэффициента риска (COR)
def calculate_cor(ecl, df):
    total_outstanding_amount = df['AMT_PRINCIPAL'].sum()
    return ecl / total_outstanding_amount

cor = calculate_cor(ecl, df_transformed)

# Шаг 8: Запись результатов в текстовый файл
def write_results_to_file(migration_matrix, pd_12m, pd_lifetime, ecl, cor, file_name='results.txt'):
    with open(file_name, 'w') as f:
        f.write("Migration Matrix:\n")
        f.write(migration_matrix.to_string())
        f.write("\n\n")
        f.write(f"PD (12-month): {pd_12m:.4f}\n")
        f.write(f"PD (Lifetime): {pd_lifetime:.4f}\n")
        f.write(f"ECL: {ecl:.2f}\n")
        f.write(f"COR: {cor:.4f}\n")

write_results_to_file(migration_matrix, pd_12m, pd_lifetime, ecl, cor)

# Визуализация матрицы миграций
def visualize_migration_matrix(migration_matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(migration_matrix, cmap='viridis', interpolation='nearest')

    plt.title('Migration Matrix')
    plt.xlabel('NEXT_DPD_GROUP')
    plt.ylabel('DPD_GROUP')

    plt.colorbar(label='Probability')

    tick_marks = np.arange(len(migration_matrix.columns))
    plt.xticks(tick_marks, migration_matrix.columns, rotation=45)
    plt.yticks(tick_marks, migration_matrix.index)

    for i in range(len(migration_matrix.index)):
        for j in range(len(migration_matrix.columns)):
            plt.text(j, i, f"{migration_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='white')

    plt.tight_layout()
    plt.show()

visualize_migration_matrix(migration_matrix)
