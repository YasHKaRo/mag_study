import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from pr6_2 import ExponentialSmoothing

class MovingAverage:
    def __init__(self, window: int = 3):
        self.window = window
        self.forecasts = []
        self.errors = []

    def fit_predict(self, data: List[float]) -> List[float]:
        n = len(data)
        self.forecasts = [0] * n
        self.errors = [0] * n

        for i in range(self.window - 1):
            self.forecasts[i] = data[i]

        for i in range(self.window - 1, n):
            forecast = np.mean(data[i - self.window + 1:i + 1])
            self.forecasts[i] = forecast
            self.errors[i] = data[i] - forecast

        return self.forecasts

    def calculate_metrics(self, data: List[float]) -> Dict[str, float]:
        errors = np.array(self.errors[self.window - 1:])
        actual = np.array(data[self.window - 1:])

        return {
            "MSE": np.mean(errors ** 2),
            "MAE": np.mean(np.abs(errors)),
            "MAPE": np.mean(np.abs(errors / actual)) * 100
        }

    def forecast_next(self, data: List[float]) -> float:
        return np.mean(data[-self.window:])


class LinearRegressionModel:
    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0
        self.forecasts = []

    def fit_predict(self, data: List[float]) -> List[float]:
        n = len(data)
        X = np.arange(n)
        Y = np.array(data)

        self.slope, self.intercept = np.polyfit(X, Y, 1)

        self.forecasts = [self.intercept + self.slope * i for i in range(n)]
        return self.forecasts

    def calculate_metrics(self, data: List[float]) -> Dict[str, float]:
        errors = np.array(data) - np.array(self.forecasts)
        return {
            "MSE": np.mean(errors ** 2),
            "MAE": np.mean(np.abs(errors)),
            "MAPE": np.mean(np.abs(errors / data)) * 100
        }

    def forecast_next(self, steps: int = 1) -> List[float]:
        return [self.intercept + self.slope * (len(self.forecasts) + i) for i in range(steps)]

def main():
    # Исходные данные
    data_stocks = [100, 102, 105, 103, 107, 110, 108, 112, 115, 113,
                   117, 120, 118, 122, 125, 123, 127, 130, 128, 132]

    # 1. Прогнозирование тремя моделями
    exp_model = ExponentialSmoothing(alpha=0.3)
    forecasts_exp = exp_model.fit_predict(data_stocks)[0]
    metrics_exp = exp_model.calculate_metrics(data_stocks)

    # Обучение моделей
    ma_model = MovingAverage(window=3)
    forecasts_ma = ma_model.fit_predict(data_stocks)
    metrics_ma = ma_model.calculate_metrics(data_stocks)

    lr_model = LinearRegressionModel()
    forecasts_lr = lr_model.fit_predict(data_stocks)
    metrics_lr = lr_model.calculate_metrics(data_stocks)

    # 2. Комбинированный прогноз
    mse_values = [metrics_exp['MSE'], metrics_ma['MSE'], metrics_lr['MSE']]
    inverse_mse = [1 / mse for mse in mse_values]
    total_inverse = sum(inverse_mse)
    weights = [inv / total_inverse for inv in inverse_mse]

    combined_forecasts = []
    for i in range(len(data_stocks)):
        combined = (weights[0] * forecasts_exp[i] +
                    weights[1] * forecasts_ma[i] +
                    weights[2] * forecasts_lr[i])
        combined_forecasts.append(combined)

    # Метрики комбинированного прогноза
    errors_combined = np.array(data_stocks) - np.array(combined_forecasts)
    metrics_combined = {
        "MSE": np.mean(errors_combined ** 2),
        "MAE": np.mean(np.abs(errors_combined)),
        "MAPE": np.mean(np.abs(errors_combined / data_stocks)) * 100
    }

    # 3. Прогноз на 5 дней вперед
    exp_next = exp_model.forecast_next()
    ma_next = ma_model.forecast_next(data_stocks)
    lr_next = lr_model.forecast_next(steps=5)

    combined_next = []
    for i in range(5):
        ma_val = ma_next if i == 0 else ma_model.forecast_next(data_stocks + combined_next[:i])
        lr_val = lr_next[i]
        combined_next.append(weights[0] * exp_next + weights[1] * ma_val + weights[2] * lr_val)

    # 4. Вывод результатов

    print("\n1. Метрики моделей:")
    print("-" * 85)
    print(f"{'Модель':<35} | {'MSE':>10} | {'MAE':>10} | {'MAPE':>10}")
    print("-" * 85)
    print(
        f"{'Эксп. сглаживание (α=0.3)':<35} | {metrics_exp['MSE']:10.4f} | {metrics_exp['MAE']:10.4f} | {metrics_exp['MAPE']:10.4f}%")
    print(
        f"{'Скользящее среднее (окно=3)':<35} | {metrics_ma['MSE']:10.4f} | {metrics_ma['MAE']:10.4f} | {metrics_ma['MAPE']:10.4f}%")
    print(
        f"{'Линейная регрессия':<35} | {metrics_lr['MSE']:10.4f} | {metrics_lr['MAE']:10.4f} | {metrics_lr['MAPE']:10.4f}%")
    print(
        f"{'Комбинированный прогноз':<35} | {metrics_combined['MSE']:10.4f} | {metrics_combined['MAE']:10.4f} | {metrics_combined['MAPE']:10.4f}%")
    print("-" * 85)

    print(f"\n2. Веса для комбинированного прогноза:")
    print(f"   Эксп. сглаживание: {weights[0]:.3f}")
    print(f"   Скользящее среднее: {weights[1]:.3f}")
    print(f"   Линейная регрессия: {weights[2]:.3f}")

    print(f"\n3. Прогноз на 5 дней вперед:")
    print("-" * 65)
    print(f"{'День':<6} | {'Эксп.сглаж.':<12} | {'Скольз.ср.':<12} | {'Лин.регр.':<12} | {'Комбинир.':<12}")
    print("-" * 65)
    for i in range(5):
        exp_val = f"{exp_next:.2f}" if i == 0 else "—"
        ma_val = f"{ma_next:.2f}" if i == 0 else "—"

        print(f"{len(data_stocks) + i + 1:<6} | "
              f"{exp_val:<12} | "
              f"{ma_val:<12} | "
              f"{lr_next[i]:<12.2f} | "
              f"{combined_next[i]:<12.2f}")
        print("-" * 65)

    # 5. Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Сравнение моделей
    axes[0, 0].plot(data_stocks, 'o-', label='Фактические', linewidth=2)
    axes[0, 0].plot(forecasts_exp, 's--', label='Эксп. сглаживание', linewidth=1)
    axes[0, 0].plot(forecasts_ma, '^:', label='Скольз. среднее', linewidth=1)
    axes[0, 0].plot(forecasts_lr, 'v-.', label='Лин. регрессия', linewidth=1)
    axes[0, 0].plot(combined_forecasts, 'd-', label='Комбинированный', linewidth=2)
    axes[0, 0].set_xlabel('День')
    axes[0, 0].set_ylabel('Цена')
    axes[0, 0].set_title('Сравнение моделей прогнозирования')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Прогноз на будущее
    future_days = np.arange(len(data_stocks), len(data_stocks) + 5)
    axes[0, 1].plot(data_stocks, 'o-', label='История', linewidth=2)
    axes[0, 1].plot(future_days, combined_next, 's--', label='Прогноз', linewidth=2)
    axes[0, 1].set_xlabel('День')
    axes[0, 1].set_ylabel('Цена')
    axes[0, 1].set_title('Прогноз на 5 дней вперед')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Ошибки прогнозов
    time_points = range(len(data_stocks))
    axes[1, 0].bar(time_points, exp_model.errors, alpha=0.5, label='Эксп. сглаживание')
    axes[1, 0].bar(time_points, ma_model.errors, alpha=0.5, label='Скольз. среднее')
    axes[1, 0].bar(time_points, np.array(data_stocks) - np.array(forecasts_lr), alpha=0.5, label='Лин. регрессия')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('День')
    axes[1, 0].set_ylabel('Ошибка')
    axes[1, 0].set_title('Ошибки прогнозов')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Веса моделей
    models = ['Эксп. сглаживание', 'Скольз. среднее', 'Лин. регрессия']
    axes[1, 1].bar(models, weights, color=['blue', 'orange', 'green'])
    axes[1, 1].set_ylabel('Вес в комбинированном прогнозе')
    axes[1, 1].set_title('Веса моделей в комбинированном прогнозе')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()