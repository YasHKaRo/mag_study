import pandas as pd                    # Работа с данными
import numpy as np                     # Численные вычисления
import matplotlib.pyplot as plt        # Визуализация
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Метрики
from typing import List, Tuple


class ExponentialSmoothing:
    def __init__(self, alpha: float = 0.3):
        if not 0 < alpha < 1:
            raise ValueError("Параметр alpha должен быть в интервале (0, 1)")
        self.alpha = alpha
        self.smoothed = []
        self.forecasts = []
        self.errors = []

    def fit_predict(self, data: List[float]) -> Tuple[List[float], List[float]]:
        n = len(data)
        self.smoothed = [0] * n
        self.forecasts = [0] * n
        self.errors = [0] * n

        self.smoothed[0] = data[0]
        self.forecasts[0] = data[0]

        for t in range(1, n):
            self.forecasts[t] = self.smoothed[t - 1]
            self.errors[t] = data[t] - self.forecasts[t]
            self.smoothed[t] = self.alpha * data[t] + (1 - self.alpha) * self.smoothed[t - 1]

        return self.forecasts, self.smoothed

    def forecast_next(self) -> float:
        if not self.smoothed:
            raise ValueError("Модель не обучена")
        return self.smoothed[-1]

    def calculate_metrics(self, data: List[float]) -> dict:
        errors = np.array(self.errors[1:])
        actual = np.array(data[1:])

        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / actual)) * 100

        return {'MSE': mse, 'MAE': mae, 'MAPE': mape}

def main():
    data_temp = [18, 20, 19, 22, 24, 23, 25, 27, 26, 28, 30, 29, 31, 33, 32]

    # 1. Поиск оптимального α
    alphas = np.arange(0.1, 1.0, 0.1)
    best_alpha = None
    best_mse = float('inf')
    results = []

    for alpha in alphas:
        model = ExponentialSmoothing(alpha=alpha)
        forecasts, smoothed = model.fit_predict(data_temp)
        metrics = model.calculate_metrics(data_temp)
        mse = metrics['MSE']
        results.append((alpha, mse))

        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    print(f"\n1. Поиск оптимального α:")
    print("=" * 40)
    print(f"{'α':>6} | {'MSE':>12}")
    print("=" * 40)
    for alpha, mse in results:
        print(f"{alpha:6.1f} | {mse:12.4f}")
    print("=" * 40)
    print(f"Оптимальное α = {best_alpha:.1f} (MSE = {best_mse:.4f})")

    # 2. Прогноз на 16-й день с оптимальным α
    model_opt = ExponentialSmoothing(alpha=best_alpha)
    forecasts_opt, smoothed_opt = model_opt.fit_predict(data_temp)
    next_forecast_opt = model_opt.forecast_next()
    print(f"\n2. Прогноз на 16-й день с α={best_alpha:.1f}: {next_forecast_opt:.2f}°C")

    # 3. Сравнение с α=0.2
    model_02 = ExponentialSmoothing(alpha=0.2)
    forecasts_02, smoothed_02 = model_02.fit_predict(data_temp)
    metrics_02 = model_02.calculate_metrics(data_temp)
    next_forecast_02 = model_02.forecast_next()

    print(f"\n3. Сравнение с α=0.2:")
    print(f"   Прогноз на 16-й день с α=0.2: {next_forecast_02:.2f}°C")
    print(f"   MSE для α=0.2: {metrics_02['MSE']:.4f}")
    print(f"   MSE для α={best_alpha:.1f}: {best_mse:.4f}")

    # 4. Графики
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # График для оптимального α
    axes[0].plot(data_temp, 'o-', label='Фактические значения', linewidth=2)
    axes[0].plot(forecasts_opt, 's--', label=f'Прогноз (α={best_alpha:.1f})', linewidth=2)
    axes[0].set_xlabel('День')
    axes[0].set_ylabel('Температура (°C)')
    axes[0].set_title(f'Прогноз с оптимальным α={best_alpha:.1f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # График для α=0.2
    axes[1].plot(data_temp, 'o-', label='Фактические значения', linewidth=2)
    axes[1].plot(forecasts_02, 's--', label='Прогноз (α=0.2)', linewidth=2)
    axes[1].set_xlabel('День')
    axes[1].set_ylabel('Температура (°C)')
    axes[1].set_title('Прогноз с α=0.2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()