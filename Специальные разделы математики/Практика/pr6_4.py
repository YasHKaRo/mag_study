import pandas as pd                    # Работа с данными
import numpy as np                     # Численные вычисления
import matplotlib.pyplot as plt        # Визуализация
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Метрики
from typing import List, Tuple

def seasonal_decomposition(data: List[float], season_length: int = 12):
    """Декомпозиция временного ряда с сезонностью"""
    n = len(data)

    # 1. Вычисление скользящего среднего для выделения тренда
    # Для четной длины окна используем центрированное скользящее среднее
    if season_length % 2 == 0:
        # Двойное скользящее среднее
        ma1 = np.convolve(data, np.ones(season_length) / season_length, mode='valid')
        ma2 = np.convolve(ma1, np.ones(2) / 2, mode='valid')
        trend = np.concatenate([
            [np.nan] * (season_length // 2 - 1),
            ma2,
            [np.nan] * (season_length // 2)
        ])
    else:
        # Простое скользящее среднее
        trend = np.convolve(data, np.ones(season_length) / season_length, mode='same')
        trend[:season_length // 2] = np.nan
        trend[-(season_length // 2):] = np.nan

    # 2. Вычисление сезонных индексов (мультипликативная модель)
    # Сначала получаем сезонную+случайную компоненту: data / trend
    seasonal_random = []
    for i in range(n - 1):
        if np.isnan(trend[i]):
            seasonal_random.append(np.nan)
        else:
            seasonal_random.append(data[i] / trend[i])

    # 3. Усреднение по сезонам для получения сезонных индексов
    seasonal_indices = []
    for i in range(season_length):
        values = []
        for j in range(i, n - 1, season_length):
            if not np.isnan(seasonal_random[j]):
                values.append(seasonal_random[j])
        seasonal_indices.append(np.mean(values) if values else 1.0)

    # Нормализация индексов (среднее = 1)
    mean_idx = np.mean(seasonal_indices)
    seasonal_indices = [idx / mean_idx for idx in seasonal_indices]

    # 4. Восстановление компонент
    seasonal_component = []
    random_component = []

    for i in range(n - 1):
        season_idx = seasonal_indices[i % season_length]
        if np.isnan(trend[i]):
            seasonal_component.append(np.nan)
            random_component.append(np.nan)
        else:
            # Сезонная компонента
            seasonal_component.append(season_idx)
            # Случайная компонента = data / (trend * seasonal)
            random_component.append(data[i] / (trend[i] * season_idx))

    return trend, seasonal_component, random_component, seasonal_indices

def main():
    data_ice = [50, 55, 65, 80, 95, 110, 120, 115, 100, 85, 70, 60,
                55, 60, 70, 85, 100, 115, 125, 120, 105, 90, 75, 65]

    season_length = 12

    # 1. Вычисление сезонных индексов
    trend, seasonal, random, indices = seasonal_decomposition(data_ice, season_length)

    print("\n1. Сезонные индексы (12 месяцев):")
    print("-" * 40)
    for i, idx in enumerate(indices):
        print(f"   Месяц {i + 1:2d}: {idx:.4f}")
    print("-" * 40)
    print(f"   Среднее: {np.mean(indices):.4f}")

    # 2. Декомпозиция ряда
    print("\n2. Декомпозиция временного ряда выполнена:")
    print("   - Трендовая компонента выделена")
    print("   - Сезонная компонента выделена")
    print("   - Случайная компонента выделена")

    # 3. Прогноз на следующие 6 месяцев
    # Прогноз тренда с помощью линейной регрессии
    trend_valid = [t for t in trend if not np.isnan(t)]
    x_trend = np.arange(len(trend_valid))
    coeff = np.polyfit(x_trend, trend_valid, 1)

    n_ahead = 6
    trend_forecast = coeff[0] * (len(trend_valid) + np.arange(n_ahead)) + coeff[1]

    # Добавляем сезонность
    seasonal_forecast = []
    for i in range(n_ahead):
        month_idx = (len(data_ice) + i) % season_length
        seasonal_forecast.append(indices[month_idx])

    # Итоговый прогноз (мультипликативная модель)
    forecast = trend_forecast * seasonal_forecast

    print(f"\n3. Прогноз на следующие {n_ahead} месяцев:")
    print("-" * 50)
    for i, (t, s, f) in enumerate(zip(trend_forecast, seasonal_forecast, forecast)):
        print(f"   Месяц {len(data_ice) + i + 1:2d}: "
              f"Тренд={t:.2f}, Сезон={s:.4f}, Прогноз={f:.2f}")

    # 4. Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Исходный ряд
    axes[0, 0].plot(data_ice, 'o-', linewidth=2, color='blue')
    axes[0, 0].set_title('Исходный временной ряд')
    axes[0, 0].set_xlabel('Месяц')
    axes[0, 0].set_ylabel('Продажи')
    axes[0, 0].grid(True, alpha=0.3)

    # Тренд
    axes[0, 1].plot(trend, 'o-', linewidth=2, color='green')
    axes[0, 1].set_title('Трендовая компонента')
    axes[0, 1].set_xlabel('Месяц')
    axes[0, 1].set_ylabel('Значение')
    axes[0, 1].grid(True, alpha=0.3)

    # Сезонность
    axes[1, 0].plot(seasonal, 'o-', linewidth=2, color='red')
    axes[1, 0].set_title('Сезонная компонента')
    axes[1, 0].set_xlabel('Месяц')
    axes[1, 0].set_ylabel('Индекс')
    axes[1, 0].grid(True, alpha=0.3)

    # Прогноз
    axes[1, 1].plot(data_ice, 'o-', label='История', linewidth=2, color='blue')
    future_months = np.arange(len(data_ice), len(data_ice) + n_ahead)
    axes[1, 1].plot(future_months, forecast, 's--', label='Прогноз', linewidth=2, color='orange')
    axes[1, 1].set_title('Прогноз на 6 месяцев вперед')
    axes[1, 1].set_xlabel('Месяц')
    axes[1, 1].set_ylabel('Продажи')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()