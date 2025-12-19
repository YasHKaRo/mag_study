import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class ExponentialSmoothing:
    """
    Класс для прогнозирования временных рядов
    методом экспоненциального сглаживания
    """

    def __init__(self, alpha: float = 0.3):
        """
        Инициализация модели

        Parameters:
        -----------
        alpha : float
            Параметр сглаживания (0 < alpha < 1)
        """
        if not 0 < alpha <= 1:
            raise ValueError("Параметр alpha должен быть в интервале (0, 1)")

        self.alpha = alpha
        self.smoothed = []
        self.forecasts = []
        self.errors = []

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
        self.smoothed = [0.0] * n
        self.forecasts = [0.0] * n
        self.errors = [0.0] * n

        # Инициализация
        self.smoothed[0] = data[0]
        self.forecasts[0] = data[0]
        self.errors[0] = 0.0

        # Итеративное вычисление
        for t in range(1, n):
            # Прогноз на текущий момент
            self.forecasts[t] = self.smoothed[t - 1]

            # Ошибка прогноза
            self.errors[t] = data[t] - self.forecasts[t]

            # Сглаживание
            self.smoothed[t] = (self.alpha * data[t] +
                                (1 - self.alpha) * self.smoothed[t - 1])

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
            raise ValueError("Модель не обучена")

        return self.smoothed[-1]

    def calculate_metrics(self, data: List[float]) -> Dict[str, float]:
        """
        Вычисление метрик качества прогноза

        Parameters:
        -----------
        data : List[float]
            Фактические значения

        Returns:
        --------
        dict
            Словарь с метриками
        """
        if not self.errors:
            raise ValueError("Модель не обучена")

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


class AdaptiveFilter:
    """
    Класс для прогнозирования временных рядов
    методом адаптивной фильтрации
    """

    def __init__(self, window_size: int = 3, alpha_r: float = 1.0):
        """
        Инициализация модели

        Parameters:
        -----------
        window_size : int
            Размер окна (число используемых наблюдений)
        alpha_r : float
            Регулировочный коэффициент
        """
        if window_size < 1:
            raise ValueError("Размер окна должен быть >= 1")

        self.window_size = window_size
        self.alpha_r = alpha_r
        self.weights = None
        self.forecasts = []
        self.errors = []
        self.weights_history = []

    def _initialize_weights(self):
        """Инициализация весовых коэффициентов"""
        self.weights = np.ones(self.window_size) / self.window_size

    def _normalize_weights(self):
        """Нормализация весовых коэффициентов"""
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights = self.weights / weight_sum

    def fit_predict(self, data: List[float]) -> Tuple[List[float], List[np.ndarray]]:
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
        weights_history : List[np.ndarray]
            История изменения весов
        """
        n = len(data)
        data_array = np.array(data)

        self.forecasts = [0.0] * n
        self.errors = [0.0] * n
        self.weights_history = []

        # Инициализация весов
        self._initialize_weights()

        # Начинаем с момента, когда есть достаточно данных
        for t in range(self.window_size, n):
            # Сохраняем текущие веса
            self.weights_history.append(self.weights.copy())

            # Получаем окно данных
            window = data_array[t - self.window_size:t]

            # Прогноз
            forecast = np.dot(self.weights, window)
            self.forecasts[t] = forecast

            # Ошибка
            error = data[t] - forecast
            self.errors[t] = error

            # Адаптация весов
            # k_g = alpha_r / sum(window^2)
            sum_squares = np.sum(window ** 2)
            if sum_squares > 0:
                k_g = self.alpha_r / sum_squares

                # Обновление весов
                self.weights = self.weights + k_g * error * window

                # Нормализация весов
                self._normalize_weights()

        return self.forecasts, self.weights_history

    def forecast_next(self, data: List[float]) -> float:
        """
        Прогноз на следующий момент времени

        Parameters:
        -----------
        data : List[float]
            Временной ряд

        Returns:
        --------
        float
            Прогнозное значение
        """
        if self.weights is None:
            raise ValueError("Модель не обучена")

        # Берем последние window_size элементов
        window = np.array(data[-self.window_size:])
        return np.dot(self.weights, window)

    def calculate_metrics(self, data: List[float]) -> Dict[str, float]:
        """
        Вычисление метрик качества прогноза

        Parameters:
        -----------
        data : List[float]
            Фактические значения

        Returns:
        --------
        dict
            Словарь с метриками
        """
        if not self.errors:
            raise ValueError("Модель не обучена")

        # Пропускаем элементы без прогноза
        errors = np.array(self.errors[self.window_size:])
        actual = np.array(data[self.window_size:])

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


def compare_methods(data: List[float], alpha: float = 0.3,
                    window_size: int = 3, alpha_r: float = 1.0):
    """
    Сравнение методов прогнозирования

    Parameters:
    -----------
    data : List[float]
        Временной ряд
    alpha : float
        Параметр сглаживания для экспоненциального сглаживания
    window_size : int
        Размер окна для адаптивной фильтрации
    alpha_r : float
        Регулировочный коэффициент для адаптивной фильтрации
    """
    print("=" * 80)
    print("СРАВНЕНИЕ МЕТОДОВ ПРОГНОЗИРОВАНИЯ ВРЕМЕННЫХ РЯДОВ")
    print("=" * 80)
    print(f"\nИсходный временной ряд: {data}")
    print(f"Длина ряда: {len(data)}")

    # Метод 1: Экспоненциальное сглаживание
    print("\n" + "-" * 80)
    print("МЕТОД 1: ЭКСПОНЕНЦИАЛЬНОЕ СГЛАЖИВАНИЕ")
    print("-" * 80)
    print(f"Параметр сглаживания α = {alpha}")

    es_model = ExponentialSmoothing(alpha=alpha)
    es_forecasts, es_smoothed = es_model.fit_predict(data)
    es_metrics = es_model.calculate_metrics(data)
    es_next = es_model.forecast_next()

    print("\nПошаговые результаты:")
    print(f"{'t':>3} | {'s(t)':>8} | {'s̃(t)':>10} | {'ŝ(t)':>10} | {'e(t)':>10}")
    print("-" * 60)
    for t in range(len(data)):
        print(f"{t:3d} | {data[t]:8.2f} | {es_smoothed[t]:10.2f} | "
              f"{es_forecasts[t]:10.2f} | {es_model.errors[t]:10.2f}")

    print(f"\nПрогноз на следующий период: {es_next:.2f}")
    print("\nМетрики качества:")
    for metric, value in es_metrics.items():
        print(f"  {metric:10s}: {value:10.4f}")

    # Метод 2: Адаптивная фильтрация
    print("\n" + "-" * 80)
    print("МЕТОД 2: АДАПТИВНАЯ ФИЛЬТРАЦИЯ")
    print("-" * 80)
    print(f"Размер окна λ = {window_size}")
    print(f"Регулировочный коэффициент α_r = {alpha_r}")

    af_model = AdaptiveFilter(window_size=window_size, alpha_r=alpha_r)
    af_forecasts, af_weights_history = af_model.fit_predict(data)
    af_metrics = af_model.calculate_metrics(data)
    af_next = af_model.forecast_next(data)

    print("\nПошаговые результаты:")
    print(f"{'t':>3} | {'s(t)':>8} | {'ŝ(t)':>10} | {'e(t)':>10} | Веса")
    print("-" * 80)
    for t in range(len(data)):
        if t >= window_size:
            weights_str = "[" + ", ".join([f"{w:.3f}" for w in
                                           af_weights_history[t - window_size]]) + "]"
            print(f"{t:3d} | {data[t]:8.2f} | {af_forecasts[t]:10.2f} | "
                  f"{af_model.errors[t]:10.2f} | {weights_str}")
        else:
            print(f"{t:3d} | {data[t]:8.2f} | {'—':>10} | {'—':>10} | —")

    print(f"\nТекущие веса: {af_model.weights}")
    print(f"Прогноз на следующий период: {af_next:.2f}")
    print("\nМетрики качества:")
    for metric, value in af_metrics.items():
        print(f"  {metric:10s}: {value:10.4f}")

    # Сравнение методов
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ МЕТОДОВ")
    print("=" * 80)
    print(f"{'Метод':<30} | {'MSE':>12} | {'MAE':>12} | {'MAPE':>12}")
    print("-" * 80)
    print(f"{'Экспоненциальное сглаживание':<30} | "
          f"{es_metrics['MSE']:12.4f} | {es_metrics['MAE']:12.4f} | "
          f"{es_metrics['MAPE']:12.4f}")
    print(f"{'Адаптивная фильтрация':<30} | "
          f"{af_metrics['MSE']:12.4f} | {af_metrics['MAE']:12.4f} | "
          f"{af_metrics['MAPE']:12.4f}")

    # Определение лучшего метода
    if es_metrics['MSE'] < af_metrics['MSE']:
        print("\n✓ Экспоненциальное сглаживание показало лучший результат")
    else:
        print("\n✓ Адаптивная фильтрация показала лучший результат")

    # Визуализация
    visualize_results(data, es_model, af_model, window_size)


def visualize_results(data: List[float], es_model: ExponentialSmoothing,
                      af_model: AdaptiveFilter, window_size: int):
    """
    Визуализация результатов прогнозирования

    Parameters:
    -----------
    data : List[float]
        Исходный временной ряд
    es_model : ExponentialSmoothing
        Модель экспоненциального сглаживания
    af_model : AdaptiveFilter
        Модель адаптивной фильтрации
    window_size : int
        Размер окна адаптивной фильтрации
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    t = np.arange(len(data))

    # График 1: Экспоненциальное сглаживание
    axes[0, 0].plot(t, data, 'o-', label='Факт', linewidth=2, markersize=6)
    axes[0, 0].plot(t, es_model.forecasts, 's--', label='Прогноз',
                    linewidth=2, markersize=5)
    axes[0, 0].plot(t, es_model.smoothed, '^:', label='Сглаженные',
                    linewidth=2, markersize=5)
    axes[0, 0].set_xlabel('Время')
    axes[0, 0].set_ylabel('Значение')
    axes[0, 0].set_title('Экспоненциальное сглаживание')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # График 2: Адаптивная фильтрация
    axes[0, 1].plot(t, data, 'o-', label='Факт', linewidth=2, markersize=6)
    axes[0, 1].plot(t[window_size:], af_model.forecasts[window_size:],
                    's--', label='Прогноз', linewidth=2, markersize=5)
    axes[0, 1].set_xlabel('Время')
    axes[0, 1].set_ylabel('Значение')
    axes[0, 1].set_title('Адаптивная фильтрация')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # График 3: Ошибки экспоненциального сглаживания
    axes[1, 0].bar(t, es_model.errors, alpha=0.7, color='red')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Время')
    axes[1, 0].set_ylabel('Ошибка')
    axes[1, 0].set_title('Ошибки: Экспоненциальное сглаживание')
    axes[1, 0].grid(True, alpha=0.3)

    # График 4: Ошибки адаптивной фильтрации
    axes[1, 1].bar(t[window_size:], af_model.errors[window_size:],
                   alpha=0.7, color='blue')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].set_xlabel('Время')
    axes[1, 1].set_ylabel('Ошибка')
    axes[1, 1].set_title('Ошибки: Адаптивная фильтрация')
    axes[1, 1].grid(True, alpha=0.3)

    # График 5: Динамика весов адаптивной фильтрации
    if af_model.weights_history:
        weights_array = np.array(af_model.weights_history)
        for i in range(window_size):
            axes[2, 0].plot(t[window_size:], weights_array[:, i],
                            label=f'w_{i + 1}', linewidth=2)
        axes[2, 0].set_xlabel('Время')
        axes[2, 0].set_ylabel('Значение веса')
        axes[2, 0].set_title('Динамика весовых коэффициентов')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

    # График 6: Сравнение методов
    es_errors_abs = np.abs(es_model.errors[1:])
    af_errors_abs = np.abs(af_model.errors[window_size:])

    methods = ['Эксп. сглаж.', 'Адапт. фильтр']
    mae_values = [np.mean(es_errors_abs), np.mean(af_errors_abs)]

    axes[2, 1].bar(methods, mae_values, alpha=0.7, color=['red', 'blue'])
    axes[2, 1].set_ylabel('MAE')
    axes[2, 1].set_title('Сравнение методов по MAE')
    axes[2, 1].grid(True, alpha=0.3, axis='y')

    # Добавление значений на столбцы
    for i, v in enumerate(mae_values):
        axes[2, 1].text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# Демонстрация работы
def main():
    """
    Основная функция для демонстрации работы моделей
    """
    # Исходные данные
    data = [150, 165, 155, 170, 180, 175, 190, 200, 195, 210, 220, 215]

    # Сравнение методов
    compare_methods(data, alpha=0.3, window_size=3, alpha_r=1.0)

    # Дополнительный анализ: влияние параметров
    print("\n" + "=" * 80)
    print("АНАЛИЗ ВЛИЯНИЯ ПАРАМЕТРОВ")
    print("=" * 80)

    # Влияние параметра alpha
    print("\nВлияние параметра α на качество прогноза:")
    print(f"{'α':>6} | {'MSE':>12} | {'MAE':>12} | {'MAPE':>12}")
    print("-" * 60)

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for alpha in alphas:
        model = ExponentialSmoothing(alpha=alpha)
        model.fit_predict(data)
        metrics = model.calculate_metrics(data)
        print(f"{alpha:6.1f} | {metrics['MSE']:12.4f} | "
              f"{metrics['MAE']:12.4f} | {metrics['MAPE']:12.4f}")

    # Влияние размера окна
    print("\nВлияние размера окна λ на качество прогноза:")
    print(f"{'λ':>6} | {'MSE':>12} | {'MAE':>12} | {'MAPE':>12}")
    print("-" * 60)

    window_sizes = [2, 3, 4, 5]
    for ws in window_sizes:
        if ws < len(data):
            model = AdaptiveFilter(window_size=ws, alpha_r=1.0)
            model.fit_predict(data)
            metrics = model.calculate_metrics(data)
            print(f"{ws:6d} | {metrics['MSE']:12.4f} | "
                  f"{metrics['MAE']:12.4f} | {metrics['MAPE']:12.4f}")


if __name__ == "__main__":
    main()
