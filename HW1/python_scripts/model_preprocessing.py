import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler


# Функция предварительной обработки данных
def preprocess_data(train_file_path, test_file_path):
    # Загрузка данных
    train_df = pd.read_csv(train_file_path)
    # Загрузка тестовых данных
    test_df = pd.read_csv(test_file_path)

    # Создание экземпляра StandardScaler
    scaler = StandardScaler()

    # Обучающий StandardScaler на основе обучающих данных
    scaler.fit(train_df[['temperature']])

    # Применение StandardScaler для обработки данных
    train_scaled_data = scaler.transform(train_df[['temperature']])
    # Применение StandardScaler к тестовым данным
    test_scaled_data = scaler.transform(test_df[['temperature']])

    # Сохранение масштабированных обучающих данных
    train_df['temperature'] = train_scaled_data
    train_df.to_csv(
        train_file_path.replace('.csv', '_preprocessed.csv'), index=False)

    # Сохранение масштабированных тестовых данных
    test_df['temperature'] = test_scaled_data
    test_df.to_csv(
        test_file_path.replace('.csv', '_preprocessed.csv'), index=False)


# Получение количества наборов данных
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # Значение по умолчанию, если аргумент не передан

for i in range(n_datasets):
    # Предварительная обработка и хранение данных для обучения и тестирования
    preprocess_data(
        f'train/temperature_train_{i+1}.csv',
        f'test/temperature_test_{i+1}.csv')
