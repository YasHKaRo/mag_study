import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from pr7_1 import ExponentialSmoothing

class HoltDoubleExponentialSmoothing:
    """
    Класс для двойного экспоненциального сглаживания (метод Хольта)
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        """
        Инициализация модели

        Parameters:
        -----------
        alpha : float
            Параметр сглаживания уровня (0 < alpha < 1)
        beta : float
            Параметр сглаживания тренда (0 < beta < 1)
        """
        if not 0 < alpha < 1 or not 0 < beta < 1:
            raise ValueError("Параметры alpha и beta должны быть в интервале (0, 1)")

        self.alpha = alpha
        self.beta = beta
        self.level = []  # Уровень L(t)
        self.trend = []  # Тренд T(t)
        self.forecasts = []  # Прогнозы
        self.errors = []  # Ошибки

    def _initialize(self, data: List[float]):
        """
        Инициализация компонент уровня и тренда
        """
        self.level = [0.0] * len(data)
        self.trend = [0.0] * len(data)

        # Инициализация по формуле: L(0) = s(0), T(0) = s(1) - s(0)
        self.level[0] = data[0]
        self.trend[0] = data[1] - data[0]

    def fit_predict(self, data: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """
        Обучение модели и получение прогнозов

        Parameters:
        -----------
        data : List[float]
            Временной ряд

        Returns:
        --------
        forecasts : List[float]
            Прогнозные значения на один шаг вперед
        level : List[float]
            Компонента уровня
        trend : List[float]
            Компонента тренда
        """
        n = len(data)
        self._initialize(data)
        self.forecasts = [0.0] * n
        self.errors = [0.0] * n

        # Первый прогноз (для t=1)
        self.forecasts[1] = self.level[0] + self.trend[0]
        self.errors[1] = data[1] - self.forecasts[1]

        # Рекурсивное вычисление для t=1..n-1
        for t in range(1, n):
            # Обновление уровня
            self.level[t] = (self.alpha * data[t] +
                             (1 - self.alpha) * (self.level[t - 1] + self.trend[t - 1]))

            # Обновление тренда
            self.trend[t] = (self.beta * (self.level[t] - self.level[t - 1]) +
                             (1 - self.beta) * self.trend[t - 1])

            # Прогноз на следующий шаг (если не последний)
            if t < n - 1:
                self.forecasts[t + 1] = self.level[t] + self.trend[t]
                self.errors[t + 1] = data[t + 1] - self.forecasts[t + 1]

        return self.forecasts, self.level, self.trend

    def forecast_future(self, steps: int = 5) -> List[float]:
        """
        Прогноз на несколько шагов вперед

        Parameters:
        -----------
        steps : int
            Число шагов прогноза

        Returns:
        --------
        List[float]
            Прогнозные значения
        """
        if not self.level or not self.trend:
            raise ValueError("Модель не обучена")

        last_level = self.level[-1]
        last_trend = self.trend[-1]

        return [last_level + k * last_trend for k in range(1, steps + 1)]

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

        # Пропускаем первые два элемента (нет прогноза)
        errors = np.array(self.errors[2:])
        actual = np.array(data[2:])

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


def compare_holt_simple(data: List[float], alpha: float = 0.3, beta: float = 0.1):
    """
    Сравнение метода Хольта с простым экспоненциальным сглаживанием
    """

    print("=" * 80)
    print("ЗАДАНИЕ 5: ПРОГНОЗИРОВАНИЕ С УЧЕТОМ ТРЕНДА (МЕТОД ХОЛЬТА)")
    print("=" * 80)

    # 1. Метод Хольта
    print("\n1. МЕТОД ДВОЙНОГО ЭКСПОНЕНЦИАЛЬНОГО СГЛАЖИВАНИЯ (ХОЛЬТ):")
    print(f"   Параметры: α = {alpha}, β = {beta}")

    holt_model = HoltDoubleExponentialSmoothing(alpha=alpha, beta=beta)
    holt_forecasts, holt_level, holt_trend = holt_model.fit_predict(data)
    holt_metrics = holt_model.calculate_metrics(data)

    # 2. Простое экспоненциальное сглаживание (для сравнения)
    print("\n2. ПРОСТОЕ ЭКСПОНЕНЦИАЛЬНОЕ СГЛАЖИВАНИЕ (ДЛЯ СРАВНЕНИЯ):")
    print(f"   Параметр: α = {alpha}")

    simple_model = ExponentialSmoothing(alpha=alpha)
    simple_forecasts, _ = simple_model.fit_predict(data)
    simple_metrics = simple_model.calculate_metrics(data)

    # 3. Прогноз на 5 шагов вперед
    print("\n3. ПРОГНОЗ НА 5 ШАГОВ ВПЕРЕД:")

    holt_future = holt_model.forecast_future(steps=5)
    simple_future = [simple_model.forecast_next()] * 5  # Для простого метода прогноз постоянный

    print(f"   Метод Хольта: {[f'{x:.2f}' for x in holt_future]}")
    print(f"   Простой метод: {[f'{x:.2f}' for x in simple_future]}")

    # 4. Сравнение методов
    print("\n4. СРАВНЕНИЕ МЕТОДОВ:")
    print(f"{'Метод':<35} | {'MSE':>12} | {'MAE':>12} | {'MAPE':>12}")
    print("-" * 80)

    print(f"{'Двойное сглаживание (Хольт)':<35} | "
          f"{holt_metrics['MSE']:12.4f} | {holt_metrics['MAE']:12.4f} | "
          f"{holt_metrics['MAPE']:12.4f}")

    print(f"{'Простое сглаживание':<35} | "
          f"{simple_metrics['MSE']:12.4f} | {simple_metrics['MAE']:12.4f} | "
          f"{simple_metrics['MAPE']:12.4f}")

    improvement = ((simple_metrics['MSE'] - holt_metrics['MSE']) /
                   simple_metrics['MSE'] * 100)
    print(f"\nУлучшение MSE при использовании метода Хольта: {improvement:.1f}%")

    # 5. Визуализация
    visualize_holt_results(data, holt_model, simple_model, holt_future, simple_future)

    return holt_model, simple_model


def visualize_holt_results(data: List[float], holt_model: HoltDoubleExponentialSmoothing,
                           simple_model, holt_future: List[float], simple_future: List[float]):
    """
    Визуализация результатов метода Хольта
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    t_data = np.arange(len(data))
    t_future = np.arange(len(data), len(data) + 5)

    # График 1: Фактические данные и прогнозы
    axes[0, 0].plot(t_data, data, 'o-', label='Факт', linewidth=2, markersize=6)
    axes[0, 0].plot(t_data, holt_model.forecasts, 's--', label='Прогноз Хольта',
                    linewidth=1.5, markersize=4)
    axes[0, 0].plot(t_data, simple_model.forecasts, '^:', label='Прогноз простой',
                    linewidth=1.5, markersize=4, alpha=0.7)
    axes[0, 0].set_xlabel('Время')
    axes[0, 0].set_ylabel('Значение')
    axes[0, 0].set_title('Сравнение прогнозов')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # График 2: Компоненты метода Хольта
    axes[0, 1].plot(t_data, holt_model.level, 'o-', label='Уровень L(t)',
                    linewidth=2, markersize=6)
    axes[0, 1].plot(t_data, holt_model.trend, 's--', label='Тренд T(t)',
                    linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Время')
    axes[0, 1].set_ylabel('Значение')
    axes[0, 1].set_title('Компоненты метода Хольта')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # График 3: Прогноз на будущие периоды
    axes[0, 2].plot(t_data, data, 'o-', label='История', linewidth=2, markersize=6)
    axes[0, 2].plot(t_future, holt_future, 'r*--', label='Прогноз Хольта',
                    linewidth=2, markersize=10)
    axes[0, 2].plot(t_future, simple_future, 'b^:', label='Прогноз простой',
                    linewidth=2, markersize=10, alpha=0.7)
    axes[0, 2].set_xlabel('Время')
    axes[0, 2].set_ylabel('Значение')
    axes[0, 2].set_title('Прогноз на 5 шагов вперед')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # График 4: Ошибки метода Хольта
    axes[1, 0].bar(t_data[2:], holt_model.errors[2:], alpha=0.7, color='red')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Время')
    axes[1, 0].set_ylabel('Ошибка')
    axes[1, 0].set_title('Ошибки метода Хольта')
    axes[1, 0].grid(True, alpha=0.3)

    # График 5: Сравнение ошибок
    holt_errors_abs = np.abs(holt_model.errors[2:])
    simple_errors_abs = np.abs(simple_model.errors[1:])

    methods = ['Метод Хольта', 'Простой метод']
    mae_values = [np.mean(holt_errors_abs), np.mean(simple_errors_abs)]

    axes[1, 1].bar(methods, mae_values, alpha=0.7, color=['blue', 'green'])
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Сравнение методов по MAE')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(mae_values):
        axes[1, 1].text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')

    # График 6: Разложение ряда
    axes[1, 2].plot(t_data, data, 'o-', label='Исходный ряд', linewidth=2, markersize=6)
    axes[1, 2].plot(t_data, holt_model.level, 's--', label='Уровень',
                    linewidth=1.5, markersize=4)
    trend_component = np.array(holt_model.level) + np.array(holt_model.trend)
    axes[1, 2].plot(t_data, trend_component, '^:', label='Уровень+Тренд',
                    linewidth=1.5, markersize=4)
    axes[1, 2].set_xlabel('Время')
    axes[1, 2].set_ylabel('Значение')
    axes[1, 2].set_title('Разложение ряда на компоненты')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """
    Основная функция для выполнения задания 5
    """
    # Исходные данные (ряд с линейным трендом)
    data = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]

    print("Исходный временной ряд:")
    print(data)
    print(f"Длина ряда: {len(data)}")

    # Сравнение методов
    holt_model, simple_model = compare_holt_simple(data, alpha=0.3, beta=0.1)

    # Вывод пошаговых расчетов
    print("\n5. ПОШАГОВЫЕ РАСЧЕТЫ МЕТОДА ХОЛЬТА:")
    print(f"{'t':>3} | {'s(t)':>8} | {'L(t)':>10} | {'T(t)':>10} | {'ŝ(t)':>10} | {'e(t)':>10}")
    print("-" * 70)

    for t in range(len(data)):
        print(f"{t:3d} | {data[t]:8.2f} | {holt_model.level[t]:10.2f} | "
              f"{holt_model.trend[t]:10.2f} | {holt_model.forecasts[t]:10.2f} | "
              f"{holt_model.errors[t]:10.2f}")

    print("\n ВЫВОД: Метод Хольта эффективно учитывает линейный тренд в данных, ")
    print("  что подтверждается меньшими ошибками прогноза по сравнению с простым")
    print("  экспоненциальным сглаживанием.")


if __name__ == "__main__":
    main()
