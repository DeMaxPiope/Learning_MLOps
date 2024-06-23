#!/bin/bash

# Функция для создания виртуальной среды
create_venv() {
    local env_name=${1:-".venv"}
    python3 -m venv "$env_name"
    echo "The virtual environment '$env_name' has been created."
}

# Функция для активации виртуальной среды
activate_venv() {
    local env_name=${1:-".venv"}
    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 create [env_name]' to create."
        return 1
    fi
    if [ -z "$VIRTUAL_ENV" ]; then
        source "./$env_name/bin/activate"
        echo "Virtual environment '$env_name' is activated."
    else
        echo "The virtual environment has already been activated."
    fi
}

# Функция для установки зависимостей от requirements.txt
install_deps() {
    if [ ! -f "requirements.txt" ]; then
        echo "File requirements.txt not found."
        return 1
    fi

    # Проверить, были ли установлены все зависимости от requirements.txt
    for package in $(cat requirements.txt | cut -d '=' -f 1); do
        if ! pip freeze | grep -q "^$package=="; then
            echo "Dependency installation..."
            pip install -r requirements.txt
            echo "Dependencies installed."
            return 0
        fi
    done

    echo "All dependencies are already installed."
}

# Создание виртуальной среды, если она еще не создана
if [ ! -d ".venv" ]; then
    create_venv
fi

# Активация виртуальной среды
activate_venv

# Установка зависимостей
install_deps

# Получение количества наборов данных
n_datasets=$1

# Запуск скрипта создания данных
python python_scripts/data_creation.py $n_datasets

# Запуск скрипта предварительной обработки данных
python python_scripts/model_preprocessing.py $n_datasets

# Запуск скрипта подготовки модели и обучения
python python_scripts/model_preparation.py $n_datasets

# Запуск скрипта тестирования модели
python python_scripts/model_testing.py $n_datasets
