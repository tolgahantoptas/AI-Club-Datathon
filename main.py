import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split, GridSearchCV

# Veri yükleme ve ön işleme
df = pd.read_csv('datasets\\ail_frx.csv')
df['date'] = pd.to_datetime(df['date'].str.strip('"').astype('int64'), unit='ms')
for column in ['open', 'high', 'low', 'close']:
    if df[column].dtype == 'O':
        df[column] = df[column].str.strip('"').astype(float)

# Eksik veriler için analiz
msno.matrix(df)
plt.show()

# Eksik 'close' değerleri için istatistiksel analiz
missing_days = df[df['close'].isnull()]
print(missing_days[['open', 'high', 'low']].describe())

# Verileri ayırma
X = df[['open', 'high', 'low']]
y = df['close'].dropna()
X_train, X_test, y_train, y_test = train_test_split(X.iloc[y.index], y, test_size=0.2, random_state=42)

# İlk model eğitimi ve MSLE hesaplama
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
msle_initial = mean_squared_log_error(y_test, y_pred)
print("Initial MSLE:", msle_initial)

# Hiperparametre ayarlama ve modeli tekrar değerlendirme
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_log_error')
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best MSLE score:", -grid_search.best_score_)

# Performans karşılaştırması
y_pred_tuned = grid_search.best_estimator_.predict(X_test)
msle_tuned = mean_squared_log_error(y_test, y_pred_tuned)
print("Tuned MSLE:", msle_tuned)

# Performans karşılaştırmasını görselleştirme
labels = ['Initial', 'Tuned']
msle_scores = [msle_initial, msle_tuned]

plt.bar(labels, msle_scores, color=['blue', 'green'])
plt.xlabel('Model Configuration')
plt.ylabel('MSLE')
plt.title('Model Performance Comparison')
plt.show()
