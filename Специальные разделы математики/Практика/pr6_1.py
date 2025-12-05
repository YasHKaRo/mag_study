import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class ExponentialSmoothing:
    """
    Класс для прогнозирования временных рядов методом экспоненциального сглаживания
    """
    def __init__(self, alpha: float = 0.3):
        """
        Инициализация модели
        Parameters:
        -----------
        alpha : float
            Параметр сглаживания (0 < alpha < 1)
        """
        if not 0 < alpha < 1:
            raise ValueError("Параметр alpha должен быть в интервале (0, 1)")
        self.alpha = alpha
        self.smoothed = []
        self.forecasts = []
        self.errors = []
        self.tracking_signal = []
    def fit_predict(self, data: List[float]) -> Tuple[List[float], List[float]]:
        """
        Обучение модели и получение прогнозов
        Parameters:
        -----------
        data : List[float]
            Временной ряд
        Returns:
        --------
        forecasts : List[float]
            Прогнозные значения
        smoothed : List[float]
            Сглаженные значения
        """
        n = len(data)
        self.smoothed = [0] * n
        self.forecasts = [0] * n
        self.errors = [0] * n
        # Инициализация
        self.smoothed[0] = data[0]
        self.forecasts[0] = data[0]
        # Вычисление сглаженных значений и прогнозов
        for t in range(1, n):
            # Прогноз на текущий момент (сделан на предыдущем шаге)
            self.forecasts[t] = self.smoothed[t-1]
            # Ошибка прогноза
            self.errors[t] = data[t] - self.forecasts[t]
            # Сглаживание
            self.smoothed[t] = self.alpha * data[t] + (1 - self.alpha) * self.smoothed[t-1]
        return self.forecasts, self.smoothed
    def forecast_next(self) -> float:
        """
        Прогноз на следующий момент времени
        Returns:
        --------
        float
            Прогнозное значение
        """
        if not self.smoothed:
            raise ValueError("Модель не обучена. Вызовите fit_predict() сначала.")
        return self.smoothed[-1]
    def calculate_metrics(self, data: List[float]) -> dict:
        """
        Вычисление метрик качества прогноза
        Parameters:
        -----------
        data : List[float]
            Фактические значения временного ряда
        Returns:
        --------
        dict
            Словарь с метриками: MSE, RMSE, MAE, MAPE
        """
        if not self.errors:
            raise ValueError("Модель не обучена. Вызовите fit_predict() сначала.")
        # Пропускаем первый элемент (нет прогноза)
        errors = np.array(self.errors[1:])
        actual = np.array(data[1:])
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / actual)) * 100
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    def calculate_tracking_signal(self) -> List[float]:
        """
        Вычисление следящего контрольного сигнала
        Returns:
        --------
        List[float]
            Значения контрольного сигнала для каждого момента времени
        """
        if not self.errors:
            raise ValueError("Модель не обучена. Вызовите fit_predict() сначала.")
        n = len(self.errors)
        smoothed_error = [0] * n
        smoothed_abs_error = [0] * n
        self.tracking_signal = [0] * n
        # Инициализация
        smoothed_error[1] = self.errors[1]
        smoothed_abs_error[1] = abs(self.errors[1])
        if smoothed_abs_error[1] != 0:
            self.tracking_signal[1] = smoothed_error[1] / smoothed_abs_error[1]
        # Вычисление контрольного сигнала
        for t in range(2, n):
            smoothed_error[t] = (self.alpha * self.errors[t] +
                                (1 - self.alpha) * smoothed_error[t-1])
            smoothed_abs_error[t] = (self.alpha * abs(self.errors[t]) +
                                     (1 - self.alpha) * smoothed_abs_error[t-1])
            if smoothed_abs_error[t] != 0:
                self.tracking_signal[t] = smoothed_error[t] / smoothed_abs_error[t]
        return self.tracking_signal
    def plot_results(self, data: List[float], title: str = "Прогнозирование временного ряда"):
        """
        Визуализация результатов прогнозирования
        Parameters:
        -----------
        data : List[float]
            Фактические значения временного ряда
        title : str
            Заголовок графика
        """
        if not self.forecasts:
            raise ValueError("Модель не обучена. Вызовите fit_predict() сначала.")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        # График 1: Фактические и прогнозные значения
        axes[0].plot(data, 'o-', label='Фактические значения', linewidth=2)
        axes[0].plot(self.forecasts, 's--', label='Прогнозные значения', linewidth=2)
        axes[0].plot(self.smoothed, '^:', label='Сглаженные значения', linewidth=2)
        axes[0].set_xlabel('Время')
        axes[0].set_ylabel('Значение')
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # График 2: Ошибки прогноза
        axes[1].bar(range(len(self.errors)), self.errors, alpha=0.7, color='red')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_xlabel('Время')
        axes[1].set_ylabel('Ошибка')
        axes[1].set_title('Ошибки прогнозирования')
        axes[1].grid(True, alpha=0.3)
        # График 3: Следящий контрольный сигнал
        if self.tracking_signal:
            axes[2].plot(self.tracking_signal, 'g-', linewidth=2)
            axes[2].axhline(y=0.55, color='red', linestyle='--', label='Верхняя граница (95%)')
            axes[2].axhline(y=-0.55, color='red', linestyle='--', label='Нижняя граница (95%)')
            axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[2].set_xlabel('Время')
            axes[2].set_ylabel('Контрольный сигнал')
            axes[2].set_title('Следящий контрольный сигнал')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
# Демонстрация работы модели
def main():
    """
    Основная функция для демонстрации работы модели
    """
    # Исходные данные
    data = [120, 135, 125, 140, 155, 150, 165, 170, 160, 175]
    print("="*70)
    print("ПРОГНОЗИРОВАНИЕ ВРЕМЕННОГО РЯДА МЕТОДОМ ЭКСПОНЕНЦИАЛЬНОГО СГЛАЖИВАНИЯ")
    print("="*70)
    print(f"\nИсходный временной ряд: {data}")
    print(f"Длина ряда: {len(data)}")
    # Создание и обучение модели
    alpha = 0.3
    model = ExponentialSmoothing(alpha=alpha)
    print(f"\nПараметр сглаживания α = {alpha}")
    print("\n" + "-"*70)
    # Прогнозирование
    forecasts, smoothed = model.fit_predict(data)
    # Вывод результатов по шагам
    print("\nПОШАГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("-"*70)
    print(f"{'t':>3} | {'s(t)':>8} | {'S̃(t)':>10} | {'ŝ(t)':>10} | {'e(t)':>10}")
    print("-"*70)
    for t in range(len(data)):
        print(f"{t+1:3d} | {data[t]:8.2f} | {smoothed[t]:10.2f} | "
              f"{forecasts[t]:10.2f} | {model.errors[t]:10.2f}")
    print("-"*70)
    # Прогноз на следующий период
    next_forecast = model.forecast_next()
    print(f"\nПрогноз на период {len(data)+1}: {next_forecast:.2f}")
    # Метрики качества
    metrics = model.calculate_metrics(data)
    print("\n" + "="*70)
    print("МЕТРИКИ КАЧЕСТВА ПРОГНОЗА:")
    print("="*70)
    for metric, value in metrics.items():
        print(f"{metric:10s}: {value:10.4f}")
    # Следящий контрольный сигнал
    tracking_signal = model.calculate_tracking_signal()
    print("\n" + "="*70)
    print("СЛЕДЯЩИЙ КОНТРОЛЬНЫЙ СИГНАЛ:")
    print("="*70)
    print(f"{'t':>3} | {'K(t)':>10} | {'Статус':>20}")
    print("-"*70)
    for t in range(1, len(tracking_signal)):
        status = "OK" if abs(tracking_signal[t]) <= 0.55 else "ТРЕВОГА!"
        print(f"{t+1:3d} | {tracking_signal[t]:10.4f} | {status:>20}")
    print("-"*70)
    # Визуализация
    model.plot_results(data)
    # Сравнение разных значений alpha
    print("\n" + "="*70)
    print("СРАВНЕНИЕ РАЗНЫХ ЗНАЧЕНИЙ ПАРАМЕТРА α:")
    print("="*70)
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []
    for a in alphas:
        m = ExponentialSmoothing(alpha=a)
        m.fit_predict(data)
        met = m.calculate_metrics(data)
        results.append({
            'alpha': a,
            'MSE': met['MSE'],
            'MAE': met['MAE'],
            'MAPE': met['MAPE']
        })
    print(f"{'α':>6} | {'MSE':>12} | {'MAE':>12} | {'MAPE':>12}")
    print("-"*70)
    for r in results:
        print(f"{r['alpha']:6.1f} | {r['MSE']:12.4f} | "
              f"{r['MAE']:12.4f} | {r['MAPE']:12.4f}")
    # Визуализация сравнения
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, a in enumerate([0.1, 0.3, 0.7, 0.9]):
        row = idx // 2
        col = idx % 2
        m = ExponentialSmoothing(alpha=a)
        f, s = m.fit_predict(data)
        axes[row, col].plot(data, 'o-', label='Факт', linewidth=2)
        axes[row, col].plot(f, 's--', label='Прогноз', linewidth=2)
        axes[row, col].set_title(f'α = {a}')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()
