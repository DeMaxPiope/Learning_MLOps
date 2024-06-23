import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import pickle

# Загрузка обучающих данных
X_train = pd.read_csv('data/train/X_train.csv')
y_train = pd.read_csv('data/train/y_train.csv')

# Создание и обучение модели случайного леса
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train['target'])

# Оценка качества модели с использованием обучающих данных
y_pred = model.predict(X_train)

# Расчет показателей качества модели
accuracy = accuracy_score(y_train['target'], y_pred)
precision = precision_score(y_train['target'], y_pred, average='weighted')
recall = recall_score(y_train['target'], y_pred, average='weighted')
f1 = f1_score(y_train['target'], y_pred, average='weighted')

# Создание DataFrame для результатов
results = pd.DataFrame({
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-score': [f1]
})

print('Results on training data:')
print(results.to_string(index=False))

# Сохранение модели в файл
os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully")
