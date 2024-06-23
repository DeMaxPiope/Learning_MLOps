import streamlit as st
import numpy as np
from joblib import load
import pickle

# Загрузка модели и масштабирующего устройства
model = load('/model/model.joblib')
with open('/model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Загрузка целевых имен из файла
with open('/model/target_names.pkl', 'rb') as f:
    target_names = pickle.load(f)

st.title('Logistic regression model on the Iris dataset')

# Ввод данных для прогнозирования
with st.container():
    sepal_length = st.slider(
        'Sepal length', min_value=0.0, max_value=10.0, value=0.0,
        step=0.1, format="%.1f")
    sepal_width = st.slider(
        'Sepal width', min_value=0.0, max_value=10.0, value=0.0,
        step=0.1, format="%.1f")
    petal_length = st.slider(
        'Petal length', min_value=0.0, max_value=10.0, value=0.0,
        step=0.1, format="%.1f")
    petal_width = st.slider(
        'Petal width', min_value=0.0, max_value=10.0, value=0.0,
        step=0.1, format="%.1f")

    if st.button('Predict'):
        sample = np.array([
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
            ]).reshape(1, -1)
        # Нормализация входных данных
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)
        st.markdown("### Prediction")
        st.markdown(
            f"The predicted species is: **{target_names[prediction[0]]}**")
