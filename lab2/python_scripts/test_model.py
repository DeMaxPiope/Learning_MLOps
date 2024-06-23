import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import pickle

# Загрузка тестовых данных
X_test = pd.read_csv('data/test/X_test.csv')
y_test = pd.read_csv('data/test/y_test.csv')

# Загрузка модели из файла
with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Прогнозирование на основе тестовых данных
y_pred = model.predict(X_test)

# Расчет показателей качества модели
accuracy = accuracy_score(y_test['target'], y_pred)
precision = precision_score(y_test['target'], y_pred, average='weighted')
recall = recall_score(y_test['target'], y_pred, average='weighted')
f1 = f1_score(y_test['target'], y_pred, average='weighted')

# Создание DataFrame для результатов
results = pd.DataFrame({
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-score': [f1]
})

print('Results on test data:')
print(results.to_string(index=False))
