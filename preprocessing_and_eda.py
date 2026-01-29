import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

# Загрузка данных
data = pd.read_csv('PaySim_data.csv')
df = pd.DataFrame(data)

print('Размер датасета:', df.shape)
print('Количество транзакции:', (len(df)))

fraud_rate = df['isFraud'].mean() * 100
print(f'Доля мошеннических операции {fraud_rate:.4f}%')
print(f"Количество мошеннических операции {df['isFraud'].sum():,}")
print(f"Количество честных операции {(df['isFraud'] == 0).sum():,}")

# Удаление лишних колонок
cleaned_df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])

print(cleaned_df.head(5))
print('Количество пропущенных значений в датасете\n', cleaned_df.isnull().sum())
print('Общая информация по датасету:')
print(cleaned_df.info())

# Группировка по всем типам транзакции
fraud_by_type = df.groupby('type')['isFraud'].sum()
print('Распределение мошеннических операции (isFraud) по каждому типу транзакции\n', fraud_by_type)

'''
По данному анализу можно понять все мошеннические операции происходят 
во время того когда мошенники переводят 
средства на другой счет (TRANSFER) и сразу их обналичивают (CASH_OUT) 
'''

# Кодирую признаки с помощью
df_filtered = pd.get_dummies(cleaned_df, columns=['type'], dtype=int)
print(df_filtered.head(5))

# Количество мошеннических и Не мошеннических операции
print('Распределение целевой переменной')
print(df_filtered['isFraud'].value_counts())

# Визуализация
sns.countplot(x='isFraud', data=df_filtered)
plt.title('Распределение мошенничества (0 = Легитимно, 1 = Мошенничество)')
plt.grid()
plt.savefig('Frauds.png')
plt.show()

# Создание признаков (Feature engineering)
# Создаю признаки для выявления технических аномалий и для облегчения работы алгоритму
# step — это часы. Извлекаем час дня (0-23)
df_filtered['hour'] = df_filtered['step'] % 24

# Признак ошибки баланса
df_filtered['errorBalanceOrig'] = df_filtered['newbalanceOrig'] + df_filtered['amount'] - df_filtered['oldbalanceOrg']
df_filtered['errorBalanceDest'] = df_filtered['oldbalanceDest'] + df_filtered['amount'] - df_filtered['newbalanceDest']

# Признаки errorBalance были созданы, чтобы сказать модели, если сумма аккаунта с новым балансом(newbalanceOrig)
# с суммой транзакции(amount) - сумма аккаунта с предыдущим балансом == 0, операция скорее всего является мошеннической

# Удаляю колонки, которые использовал для создания метки ошибок
df_final = df_filtered.drop(columns=['step', 'oldbalanceOrg',  'newbalanceOrig',  'oldbalanceDest', 'newbalanceDest'])

plt.figure(figsize=(12, 8))
sns.heatmap(df_final.corr(), fmt='.2f', cmap='RdBu', annot=True)
plt.title('Корреляционный анализ между признаками с новыми колонками ошибок')
plt.savefig('payments_corr_heatmapv2.png')
plt.show()

print(df_final.head(5))

df_final.to_csv('Updated_Paysim_data.csv', index=False)
print("Данные сохранены в processed_data.csv")

import joblib
features = df_final.drop(columns=['isFraud']).columns.tolist()

joblib.dump(features, 'model_features.pkl')
print(f"Список признаков сохранен: {features}")

full_structure = df_final.columns.tolist()
joblib.dump(full_structure, 'final_df_columns.pkl')
print("Артефакты предобработки зафиксированы.")