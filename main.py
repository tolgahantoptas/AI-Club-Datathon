import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

df = pd.read_csv('datasets\\ail_frx.csv')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

df.head()
# Veri Setini Düzenleme

# Tarih sütununu düzenleme
if df['date'].dtype == 'O':  # 'O' pandas'da object, yani string türünü belirtir
    df['date'] = pd.to_datetime(df['date'].str.strip('"').astype('int64'), unit='ms')

# Sayısal sütunları düzenleme
for column in ['open', 'high', 'low', 'close']:
    if df[column].dtype == 'O':  # Sadece string türündeki sütunlar üzerinde işlem yap
        df[column] = df[column].str.strip('"').astype(float)

########## EDA ############
# Eksik veriler
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# Eksik veri matrisi
msno.matrix(df)
plt.show()  # korelasyonlar yüksek

# Ay bazında eksik 'close' değerlerinin sayısını hesaplama
monthly_missing = df.set_index('date')['close'].isnull().resample('M').sum()
monthly_missing.plot(kind='bar', color='skyblue')
plt.title('Ay Bazında Eksik "Close" Değerleri')
plt.ylabel('Eksik Değer Sayısı')
plt.xlabel('Tarih')
plt.show()

# Eksik 'close' değerleri olan günlerde diğer finansal göstergelerin istatistiklerini göster
missing_days = df[df['close'].isnull()]
print(missing_days[['open', 'high', 'low']].describe())

df['date'] = pd.to_datetime(df['date'])

# Eksik 'close' değerlerine sahip günlerin verilerini alın
missing_close_days = df[df['close'].isnull()]

# 'open', 'high', 'low' sütunları arasındaki korelasyon matrisini hesaplayın
correlation_matrix = missing_close_days[['open', 'high', 'low']].corr()

#           open      high       low
# open  1.000000  0.999966  0.999971
# high  0.999966  1.000000  0.999935
# low   0.999971  0.999935  1.000000

from sklearn.linear_model import SGDRegressor

# Modeli SGDRegressor ile eğit
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.01)
sgd_model.fit(X_train, y_train)

# Eksik close değerlerini tahmin et
predicted_close_sgd = sgd_model.predict(X_missing)

# Eksik değerleri doldur
df.loc[df['close'].isnull(), 'close'] = predicted_close_sgd

# Close sütununun dağılımını incele
plt.figure(figsize=(10, 6))
sns.histplot(df['close'], kde=True)
plt.title('Close Değerlerinin Dağılımı')
plt.xlabel('Close')
plt.ylabel('Frekans')
plt.show()


# Close değerlerinin zaman serisi grafiği
df.set_index('date')['close'].plot(figsize=(15, 7))
plt.title('Zaman İçinde Close Değerleri')
plt.xlabel('Tarih')
plt.ylabel('Close')
plt.show()

# Base Model
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Verileri hazırlama
X = df[['open', 'high', 'low']]
y = df['close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğitme
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# MSLE hesaplama
msle_initial = mean_squared_log_error(y_test, y_pred)
print("Initial MSLE:", msle_initial)
# Initial MSLE: 6.101201479451752e-07




