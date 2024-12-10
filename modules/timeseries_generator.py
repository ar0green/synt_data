import numpy as np
import pandas as pd

def generate_timeseries(num_points=50, trend=1.0, noise_level=0.1, seasonality=0.0):
    """
    Генерирует временной ряд вида: y = trend * x + noise + seasonality_component
    Параметры:
        num_points: количество точек
        trend: сила линейного тренда
        noise_level: уровень шума (стандартное отклонение)
        seasonality: амплитуда сезонной компоненты (синусоидальной)
    """
    x = np.arange(num_points)
    noise = noise_level * np.random.randn(num_points)
    seasonal = seasonality * np.sin(2 * np.pi * x / (num_points/5))
    y = trend * x + noise + seasonal
    
    df = pd.DataFrame({"time": x, "value": y})
    return df
