from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pickle

# Загрузка набора данных
iris = load_iris()
X = iris.data  # type: ignore
y = iris.target  # type: ignore

# Разделение набора данных на обучающий и тестовый
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True)

# Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создание и обучение модели
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Сохранение модели и масштабирующего устройства
dump(model, 'model.joblib')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Сохранение имен целевых объектов в файл
with open('target_names.pkl', 'wb') as f:
    pickle.dump(iris.target_names, f)  # type: ignore
