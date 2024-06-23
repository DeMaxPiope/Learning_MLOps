import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib
import sys


# Функция для тестирования модели
def test_model(model_path, test_data_path):
    # Загрузка обученной модели
    model = joblib.load(model_path)

    # Загрузка тестовых данных
    df_test = pd.read_csv(test_data_path)

    # Разделение данных на объекты и целевую переменную
    X_test = df_test[['temperature']]
    y_test = df_test['anomaly']

    # Прогнозирование на основе тестовых данных
    y_pred = model.predict(X_test)

    # Расчет показателей
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Создание DataFrame для результатов
    results = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1]
    })

    return results


# Получение количества наборов данных
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # Значение по умолчанию, если аргумент не передан
print()
for i in range(n_datasets):
    # Путь к обученной модели
    model_path = f'models/model_{i+1}.pkl'
    # Путь к тестовым данным
    test_data_path = f'test/temperature_test_{i+1}_preprocessed.csv'

    # Тестирование модели
    results = test_model(model_path, test_data_path)
    print(f"The model for dataset {i+1} is tested.")
    print(results.to_string(index=False))
    print('-' * 20)
