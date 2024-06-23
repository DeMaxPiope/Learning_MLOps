import pandas as pd
import os
from sklearn.datasets import load_wine

# Загрузка данных
wine = load_wine()
X = wine.data  # type: ignore
y = wine.target  # type: ignore

# Преобразование данных в DataFrame
df = pd.DataFrame(data=X, columns=wine.feature_names)  # type: ignore
df['target'] = y

print(df.info())
print(df.describe())

# Создание каталогов для хранения данных
os.makedirs('data', exist_ok=True)

# Сохранение данных в CSV-файлы
df.to_csv('data/wine.csv', index=False)
