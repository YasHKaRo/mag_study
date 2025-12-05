import pandas as pd                    # Работа с данными
import numpy as np                     # Численные вычисления
import matplotlib.pyplot as plt        # Визуализация
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Метрики
from typing import List, Tuple
from pr6_2 import ExponentialSmoothing

class HoltDoubleExponentialSmoothing:
    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        if not 0 < alpha < 1 or not 0 < beta < 1:
            raise ValueError("Параметры должны быть в интервале (0, 1)")
        self.alpha = alpha
        self.beta = beta
        self.S = []  # уровень
        self.T = []  # тренд
        self.forecasts = []
        self.errors = []

    def fit_predict(self, data: List[float]) -> Tuple[List[float], List[float]]:
        n = len(data)
        self.S = [0] * n
        self.T = [0] * n
        self.forecasts = [0] * n
        self.errors = [0] * n

        # Инициализация как в подсказке
        self.S[0] = data[0]
        self.T[0] = data[1] - data[0]  # начальный тренд
        self.forecasts[0] = data[0]

        for t in range(1, n):
            # Прогноз на текущий момент
            self.forecasts[t] = self.S[t - 1] + self.T[t - 1]

            # Ошибка
            self.errors[t] = data[t] - self.forecasts[t]

            # Обновление уровня
            self.S[t] = (self.alpha * data[t] +
                         (1 - self.alpha) * (self.S[t - 1] + self.T[t - 1]))

            # Обновление тренда
            self.T[t] = (self.beta * (self.S[t] - self.S[t - 1]) +
                         (1 - self.beta) * self.T[t - 1])

        return self.forecasts, self.S

    def forecast_next(self, k: int = 1) -> float:
        if not self.S or not self.T:
            raise ValueError("Модель не обучена")
        return self.S[-1] + k * self.T[-1]

    def calculate_metrics(self, data: List[float]) -> dict:
        errors = np.array(self.errors[1:])
        actual = np.array(data[1:])

        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / actual)) * 100

        return {'MSE': mse, 'MAE': mae, 'MAPE': mape}

def main():

    data_sales = [100, 110, 115, 125, 135, 140, 150, 160, 165, 175, 185, 190]
    alpha = 0.3
    beta = 0.1

    # 1. Метод Хольта
    model_holt = HoltDoubleExponentialSmoothing(alpha=alpha, beta=beta)
    forecasts_holt, S_holt = model_holt.fit_predict(data_sales)
    metrics_holt = model_holt.calculate_metrics(data_sales)

    print(f"\n1. Метод двойного экспоненциального сглаживания (Хольт):")
    print(f"   α = {alpha}, β = {beta}")
    print(f"   MSE = {metrics_holt['MSE']:.4f}")

    # 2. Прогноз на 3 месяца вперед
    print(f"\n2. Прогноз на следующие 3 месяца:")
    for k in range(1, 4):
        forecast = model_holt.forecast_next(k)
        print(f"   Месяц {len(data_sales) + k}: {forecast:.2f}")

    # 3. Сравнение с простым экспоненциальным сглаживанием
    model_simple = ExponentialSmoothing(alpha=alpha)
    forecasts_simple, smoothed_simple = model_simple.fit_predict(data_sales)
    metrics_simple = model_simple.calculate_metrics(data_sales)

    print(f"\n3. Сравнение с простым экспоненциальным сглаживанием:")
    print(f"   MSE (Хольт): {metrics_holt['MSE']:.4f}")
    print(f"   MSE (простое): {metrics_simple['MSE']:.4f}")
    print(f"   Разница: {abs(metrics_holt['MSE'] - metrics_simple['MSE']):.4f}")

    # 4. Графики
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # График метода Хольта
    axes[0].plot(data_sales, 'o-', label='Фактические значения', linewidth=2)
    axes[0].plot(forecasts_holt, 's--', label='Прогноз (Хольт)', linewidth=2)
    axes[0].set_xlabel('Месяц')
    axes[0].set_ylabel('Продажи')
    axes[0].set_title('Метод двойного экспоненциального сглаживания (Хольт)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # График простого сглаживания
    axes[1].plot(data_sales, 'o-', label='Фактические значения', linewidth=2)
    axes[1].plot(forecasts_simple, 's--', label='Прогноз (простое)', linewidth=2)
    axes[1].set_xlabel('Месяц')
    axes[1].set_ylabel('Продажи')
    axes[1].set_title('Простое экспоненциальное сглаживание')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()