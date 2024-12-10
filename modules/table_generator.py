import numpy as np
import pandas as pd
from faker import Faker

def generate_table_data_advanced(num_samples=10, columns_config=None, locale='en_US', correlations=None):
    """
    Генерирует табличные данные на основе подробной конфигурации колонок и опциональных корреляций.
    
    Параметры:
        num_samples: количество строк
        columns_config: список словарей с описанием столбцов
        locale: локаль Faker для категориальных данных
        correlations: матрица корреляций для числовых нормальных столбцов 
                      (двумерный список или np.array размером NxN, где N — число нормальных числовых столбцов)
                      Если None, корреляции не используются.
    """
    fake = Faker(locale)
    df = pd.DataFrame()

    # нормальные числовые, остальные числовые, категориальные
    normal_numeric_cols = []
    other_numeric_cols = []
    categorical_cols = []

    for col_conf in columns_config:
        if col_conf["type"] == "numeric":
            if col_conf["dist"] == "normal":
                normal_numeric_cols.append(col_conf)
            else:
                other_numeric_cols.append(col_conf)
        else:
            categorical_cols.append(col_conf)

    if normal_numeric_cols and correlations is not None and len(normal_numeric_cols) > 1:
        n = len(normal_numeric_cols)
        corr_matrix = np.array(correlations)
        if corr_matrix.shape != (n, n):
            raise ValueError("Размер матрицы корреляций не соответствует количеству нормальных столбцов.")
        
        # Собираем вектор средних и дисперсий
        means = []
        stds = []
        for c in normal_numeric_cols:
            means.append(c.get("mean", 0.0))
            stds.append(c.get("std", 1.0))
        
        means = np.array(means)
        stds = np.array(stds)

        # Из корреляций формируем ковариационную матрицу: Cov = D * Corr * D, где D — диагональная матрица stds
        D = np.diag(stds)
        cov_matrix = D @ corr_matrix @ D

        # Генерируем данные
        data_normal = np.random.multivariate_normal(means, cov_matrix, size=num_samples)
        
        for i, c in enumerate(normal_numeric_cols):
            df[c["name"]] = data_normal[:, i]
    else:
        # Если нет корреляций или один столбец нормальный, генерируем их по отдельности
        for c in normal_numeric_cols:
            mean = c.get("mean", 0.0)
            std = c.get("std", 1.0)
            data = np.random.randn(num_samples) * std + mean
            df[c["name"]] = data

    # Генерируем остальные числовые столбцы (uniform)
    for c in other_numeric_cols:
        dist = c.get("dist", "normal")
        if dist == "uniform":
            min_val = c.get("min_val", 0.0)
            max_val = c.get("max_val", 1.0)
            data = np.random.rand(num_samples) * (max_val - min_val) + min_val
            df[c["name"]] = data
        else:
            # fallback на normal
            mean = c.get("mean", 0.0)
            std = c.get("std", 1.0)
            df[c["name"]] = np.random.randn(num_samples)*std + mean

    # Генерируем категориальные столбцы
    for c in categorical_cols:
        faker_type = c.get("faker_type")
        categories = c.get("categories")
        if categories:
            data = np.random.choice(categories, size=num_samples)
        else:
            # Генерируем через Faker
            if faker_type == "name":
                data = [fake.name() for _ in range(num_samples)]
            elif faker_type == "city":
                data = [fake.city() for _ in range(num_samples)]
            elif faker_type == "company":
                data = [fake.company() for _ in range(num_samples)]
            else:
                data = [fake.name() for _ in range(num_samples)]
        df[c["name"]] = data
    
    return df
