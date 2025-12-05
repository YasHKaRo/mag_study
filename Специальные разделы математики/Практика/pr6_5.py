import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from scipy import stats
from pr6_2 import ExponentialSmoothing


class AR1Model:
    """Модель авторегрессии первого порядка AR(1)"""

    def __init__(self):
        self.phi0 = 0.0  # константа
        self.phi1 = 0.0  # коэффициент при s(t-1)
        self.predictions = []
        self.errors = []
        self.residuals = []

    def fit(self, data: List[float]) -> Dict[str, float]:
        """Оценка параметров методом наименьших квадратов"""
        # Для AR(1) используем пары (s(t-1), s(t))
        X = []  # s(t-1)
        Y = []  # s(t)

        for i in range(1, len(data)):
            X.append(data[i - 1])
            Y.append(data[i])

        # Преобразуем в массивы numpy
        X_arr = np.array(X)
        Y_arr = np.array(Y)

        # Оценка параметров линейной регрессии: s(t) = phi0 + phi1 * s(t-1)
        # Используем метод наименьших квадратов через numpy
        A = np.vstack([X_arr, np.ones(len(X_arr))]).T
        self.phi1, self.phi0 = np.linalg.lstsq(A, Y_arr, rcond=None)[0]

        # Вычисляем предсказания и ошибки
        self.predictions = [data[0]]  # первый элемент предсказать не можем
        self.errors = [0.0]
        self.residuals = []

        for i in range(1, len(data)):
            prediction = self.phi0 + self.phi1 * data[i - 1]
            self.predictions.append(prediction)
            error = data[i] - prediction
            self.errors.append(error)
            self.residuals.append(error)  # для проверки автокорреляции

        return {"phi0": self.phi0, "phi1": self.phi1}

    def forecast(self, data: List[float], steps: int = 1) -> List[float]:
        """Прогноз на steps шагов вперед"""
        forecasts = []
        last_value = data[-1]

        for i in range(steps):
            next_forecast = self.phi0 + self.phi1 * last_value
            forecasts.append(next_forecast)
            last_value = next_forecast

        return forecasts

    def calculate_metrics(self, data: List[float]) -> Dict[str, float]:
        """Вычисление метрик качества"""
        # Пропускаем первый элемент (нет прогноза)
        errors = np.array(self.errors[1:])
        actual = np.array(data[1:])

        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / actual)) * 100

        return {"MSE": mse, "MAE": mae, "MAPE": mape}

    def check_autocorrelation(self, max_lag: int = 5) -> Dict[int, float]:
        """Проверка остатков на автокорреляцию (критерий Дарбина-Ватсона)"""
        residuals = np.array(self.residuals)
        n = len(residuals)

        # Вычисление статистики Дарбина-Ватсона
        dw_numerator = np.sum((residuals[1:] - residuals[:-1]) ** 2)
        dw_denominator = np.sum(residuals ** 2)
        dw_statistic = dw_numerator / dw_denominator

        # Вычисление автокорреляций для разных лагов
        autocorrelations = {}
        for lag in range(1, max_lag + 1):
            if lag < n:
                corr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                autocorrelations[lag] = corr

        return {"DW": dw_statistic, "autocorrelations": autocorrelations}

def main():


    data_ar = [10, 12, 15, 14, 18, 20, 19, 23, 25, 24, 28, 30]

    # 1. Построение модели AR(1)
    ar_model = AR1Model()
    params = ar_model.fit(data_ar)

    print("\n1. Модель авторегрессии первого порядка AR(1):")
    print("-" * 50)
    print(f"   Уравнение: s(t) = {params['phi0']:.4f} + {params['phi1']:.4f} * s(t-1)")
    print(f"   phi0 (константа) = {params['phi0']:.4f}")
    print(f"   phi1 (коэффициент) = {params['phi1']:.4f}")

    # 2. Метрики качества
    metrics_ar = ar_model.calculate_metrics(data_ar)
    print("\n2. Метрики качества модели AR(1):")
    print("-" * 50)
    print(f"   MSE: {metrics_ar['MSE']:.4f}")
    print(f"   MAE: {metrics_ar['MAE']:.4f}")
    print(f"   MAPE: {metrics_ar['MAPE']:.4f}%")

    # 3. Проверка адекватности модели (автокорреляция остатков)
    autocorr_test = ar_model.check_autocorrelation(max_lag=3)
    print("\n3. Проверка остатков на автокорреляцию:")
    print("-" * 50)
    print(f"   Статистика Дарбина-Ватсона: {autocorr_test['DW']:.4f}")
    print("   Интерпретация:")
    print("   - DW ≈ 2: нет автокорреляции")
    print("   - DW < 2: положительная автокорреляция")
    print("   - DW > 2: отрицательная автокорреляция")

    print("\n   Автокорреляции остатков:")
    for lag, corr in autocorr_test['autocorrelations'].items():
        print(f"   Лаг {lag}: {corr:.4f}")

    # 4. Прогноз на 3 шага вперед
    forecast_steps = 3
    forecasts_ar = ar_model.forecast(data_ar, steps=forecast_steps)
    print(f"\n4. Прогноз на {forecast_steps} шага вперед:")
    print("-" * 50)
    for i, forecast in enumerate(forecasts_ar, 1):
        print(f"   Шаг {i}: {forecast:.2f}")

    # 5. Сравнение с экспоненциальным сглаживанием
    exp_model = ExponentialSmoothing(alpha=0.3)
    forecasts_exp, smoothed_exp = exp_model.fit_predict(data_ar)
    metrics_exp = exp_model.calculate_metrics(data_ar)

    print("\n5. Сравнение с экспоненциальным сглаживанием (α=0.3):")
    print("-" * 70)
    print(f"{'Метод':<25} | {'MSE':>10} | {'MAE':>10} | {'MAPE':>10}")
    print("-" * 70)
    print(f"{'AR(1)':<25} | {metrics_ar['MSE']:10.4f} | {metrics_ar['MAE']:10.4f} | {metrics_ar['MAPE']:10.4f}%")
    print(f"{'Экспоненциальное сглаживание':<25} | {metrics_exp['MSE']:10.4f} | {metrics_exp['MAE']:10.4f} | {metrics_exp['MAPE']:10.4f}%")
    print("-" * 70)

    if metrics_ar['MSE'] < metrics_exp['MSE']:
        print("   Вывод: AR(1) показывает лучшее качество по MSE")
    else:
        print("   Вывод: Экспоненциальное сглаживание показывает лучшее качество по MSE")

    # Визуализация для задания 5
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Исходный ряд и прогнозы AR(1)
    axes[0, 0].plot(data_ar, 'o-', label='Фактические значения', linewidth=2)
    axes[0, 0].plot(ar_model.predictions, 's--', label='Прогноз AR(1)', linewidth=2)
    axes[0, 0].set_xlabel('Время')
    axes[0, 0].set_ylabel('Значение')
    axes[0, 0].set_title('Модель AR(1): фактические и прогнозные значения')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Остатки AR(1)
    axes[0, 1].bar(range(len(ar_model.residuals)), ar_model.residuals, alpha=0.7, color='red')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].set_xlabel('Время')
    axes[0, 1].set_ylabel('Остатки')
    axes[0, 1].set_title('Остатки модели AR(1)')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Сравнение с экспоненциальным сглаживанием
    axes[1, 0].plot(data_ar, 'o-', label='Фактические значения', linewidth=2)
    axes[1, 0].plot(ar_model.predictions, 's--', label='AR(1)', linewidth=2)
    axes[1, 0].plot(forecasts_exp, '^:', label='Эксп. сглаживание', linewidth=2)
    axes[1, 0].set_xlabel('Время')
    axes[1, 0].set_ylabel('Значение')
    axes[1, 0].set_title('Сравнение AR(1) и экспоненциального сглаживания')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Прогноз на будущие периоды
    future_points = np.arange(len(data_ar), len(data_ar) + forecast_steps)
    axes[1, 1].plot(data_ar, 'o-', label='История', linewidth=2)
    axes[1, 1].plot(future_points, forecasts_ar, 's--', label='Прогноз AR(1)', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Время')
    axes[1, 1].set_ylabel('Значение')
    axes[1, 1].set_title(f'Прогноз AR(1) на {forecast_steps} шага вперед')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()