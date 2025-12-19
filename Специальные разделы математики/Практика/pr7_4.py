import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from pr7_1 import AdaptiveFilter

class AdaptiveFilterVariableWindow:
    """
    Класс для адаптивной фильтрации с переменным размером окна
    """

    def __init__(self, possible_windows: List[int] = [2, 3, 4, 5],
                 eval_window_size: int = 5, alpha_r: float = 1.0):
        """
        Инициализация модели

        Parameters:
        -----------
        possible_windows : List[int]
            Возможные размеры окон для выбора
        eval_window_size : int
            Размер окна для вычисления скользящей MSE
        alpha_r : float
            Регулировочный коэффициент
        """
        self.possible_windows = possible_windows
        self.eval_window_size = eval_window_size
        self.alpha_r = alpha_r

        self.current_window = None
        self.weights = None
        self.forecasts = []
        self.errors = []
        self.window_history = []  # История выбранных размеров окон
        self.weights_history = []
        self.mse_history = []  # История MSE для разных окон

    def _initialize_weights(self, window_size: int):
        """Инициализация весовых коэффициентов"""
        self.weights = np.ones(window_size) / window_size

    def _normalize_weights(self):
        """Нормализация весовых коэффициентов"""
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights = self.weights / weight_sum

    def _calculate_window_mse(self, errors: List[float], start_idx: int, end_idx: int) -> float:
        """
        Вычисление MSE на заданном окне ошибок

        Parameters:
        -----------
        errors : List[float]
            Список ошибок
        start_idx : int
            Начальный индекс
        end_idx : int
            Конечный индекс

        Returns:
        --------
        float
            MSE на окне
        """
        if end_idx <= start_idx or end_idx > len(errors):
            return float('inf')

        window_errors = errors[start_idx:end_idx]
        return np.mean(np.array(window_errors) ** 2)

    def _select_best_window(self, t: int, data: np.ndarray, errors: List[float]) -> int:
        """
        Выбор оптимального размера окна на основе скользящей MSE

        Parameters:
        -----------
        t : int
            Текущий момент времени
        data : np.ndarray
            Временной ряд
        errors : List[float]
            История ошибок

        Returns:
        --------
        int
            Оптимальный размер окна
        """
        # Для первых шагов используем минимальное окно
        if t < max(self.possible_windows) + self.eval_window_size:
            return min(self.possible_windows)

        best_window = self.possible_windows[0]
        best_mse = float('inf')
        window_mses = {}

        for window_size in self.possible_windows:
            # Пропускаем, если недостаточно данных
            if t < window_size + self.eval_window_size:
                continue

            # Вычисляем MSE на последних eval_window_size ошибках
            # для данной window_size
            mse_window = self._calculate_window_mse(
                errors, t - self.eval_window_size, t)

            window_mses[window_size] = mse_window

            if mse_window < best_mse:
                best_mse = mse_window
                best_window = window_size

        # Сохраняем историю MSE
        self.mse_history.append(window_mses)

        return best_window

    def fit_predict(self, data: List[float]) -> Tuple[List[float], List[int], List[Dict]]:
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
        window_history : List[int]
            История выбранных размеров окон
        mse_history : List[Dict]
            История MSE для разных окон
        """
        n = len(data)
        data_array = np.array(data)

        self.forecasts = [0.0] * n
        self.errors = [0.0] * n
        self.window_history = []
        self.weights_history = []
        self.mse_history = []

        # Инициализация для первых шагов
        init_window = min(self.possible_windows)
        self.current_window = init_window
        self._initialize_weights(init_window)

        for t in range(n):
            if t < init_window:
                # Для первых точек прогноза нет
                self.forecasts[t] = data[t]
                self.errors[t] = 0.0
                self.window_history.append(init_window)
                continue

            # 1. Выбор оптимального размера окна
            best_window = self._select_best_window(t, data_array, self.errors)

            # Если изменился размер окна, переинициализируем веса
            if best_window != self.current_window:
                self.current_window = best_window
                self._initialize_weights(self.current_window)

            self.window_history.append(self.current_window)

            # 2. Получение окна данных
            window_data = data_array[t - self.current_window:t]

            # 3. Прогноз
            forecast = np.dot(self.weights, window_data)
            self.forecasts[t] = forecast

            # 4. Ошибка
            error = data[t] - forecast
            self.errors[t] = error

            # 5. Адаптация весов (только если есть достаточно данных)
            if t >= self.current_window:
                # Сохраняем текущие веса
                self.weights_history.append(self.weights.copy())

                # Адаптация весов
                sum_squares = np.sum(window_data ** 2)
                if sum_squares > 0:
                    k_g = self.alpha_r / sum_squares
                    self.weights = self.weights + k_g * error * window_data
                    self._normalize_weights()

        return self.forecasts, self.window_history, self.mse_history

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

        window = np.array(data[-self.current_window:])
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

        # Пропускаем первые элементы без прогноза
        start_idx = min(self.possible_windows)
        errors = np.array(self.errors[start_idx:])
        actual = np.array(data[start_idx:])

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


def compare_with_fixed_window(data: List[float]):
    """
    Сравнение адаптивной фильтрации с переменным и фиксированным окном
    """
    print("=" * 80)
    print("ЗАДАНИЕ 4: АДАПТИВНАЯ ФИЛЬТРАЦИЯ С ПЕРЕМЕННЫМ ОКНОМ")
    print("=" * 80)

    # 1. Модель с переменным окном
    print("\n1. АДАПТИВНАЯ ФИЛЬТРАЦИЯ С ПЕРЕМЕННЫМ ОКНОМ:")
    print("   Возможные размеры окон: λ = [2, 3, 4, 5]")
    print("   Оценка на окне: 5 последних наблюдений")

    var_window_model = AdaptiveFilterVariableWindow(
        possible_windows=[2, 3, 4, 5],
        eval_window_size=5,
        alpha_r=1.0
    )

    var_forecasts, window_history, mse_history = var_window_model.fit_predict(data)
    var_metrics = var_window_model.calculate_metrics(data)

    print(f"\n   История выбора окон (последние 10 значений):")
    print(f"   {window_history[-10:]}")

    # 2. Модель с фиксированным окном λ = 3
    print("\n2. АДАПТИВНАЯ ФИЛЬТРАЦИЯ С ФИКСИРОВАННЫМ ОКНОМ (λ = 3):")

    # Используем класс из задания 1
    fixed_model = AdaptiveFilter(window_size=3, alpha_r=1.0)
    fixed_forecasts, _ = fixed_model.fit_predict(data)
    fixed_metrics = fixed_model.calculate_metrics(data)

    # 3. Сравнение
    print("\n3. СРАВНЕНИЕ МЕТОДОВ:")
    print(f"{'Метод':<35} | {'MSE':>12} | {'MAE':>12} | {'MAPE':>12}")
    print("-" * 80)

    print(f"{'Переменное окно (адаптивное)':<35} | "
          f"{var_metrics['MSE']:12.4f} | {var_metrics['MAE']:12.4f} | "
          f"{var_metrics['MAPE']:12.4f}")

    print(f"{'Фиксированное окно (λ=3)':<35} | "
          f"{fixed_metrics['MSE']:12.4f} | {fixed_metrics['MAE']:12.4f} | "
          f"{fixed_metrics['MAPE']:12.4f}")

    improvement = (fixed_metrics['MSE'] - var_metrics['MSE']) / fixed_metrics['MSE'] * 100
    print(f"\n Улучшение MSE при адаптивном выборе окна: {improvement:.1f}%")

    # 4. Визуализация
    visualize_variable_window_results(data, var_window_model, fixed_model, window_history, mse_history)

    return var_window_model, fixed_model, window_history


def visualize_variable_window_results(data: List[float], var_model: AdaptiveFilterVariableWindow,
                                      fixed_model: AdaptiveFilter,
                                      window_history: List[int], mse_history: List[Dict]):
    """
    Визуализация результатов адаптивной фильтрации с переменным окном
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    t = np.arange(len(data))

    # График 1: Фактические данные и прогнозы
    axes[0, 0].plot(t, data, 'o-', label='Факт', linewidth=2, markersize=6)
    axes[0, 0].plot(t, var_model.forecasts, 's--', label='Прогноз (переменное окно)',
                    linewidth=1.5, markersize=4)
    axes[0, 0].set_xlabel('Время')
    axes[0, 0].set_ylabel('Значение')
    axes[0, 0].set_title('Адаптивная фильтрация с переменным окном')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # График 2: Сравнение с фиксированным окном
    axes[0, 1].plot(t, data, 'o-', label='Факт', linewidth=2, markersize=6)
    axes[0, 1].plot(t, fixed_model.forecasts, 's--', label='Прогноз (фиксированное окно λ=3)',
                    linewidth=1.5, markersize=4, alpha=0.7)
    axes[0, 1].plot(t, var_model.forecasts, '^:', label='Прогноз (переменное окно)',
                    linewidth=1.5, markersize=4, alpha=0.7)
    axes[0, 1].set_xlabel('Время')
    axes[0, 1].set_ylabel('Значение')
    axes[0, 1].set_title('Сравнение прогнозов')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # График 3: Динамика выбора размера окна
    axes[1, 0].step(t, window_history, 'o-', where='post', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Время')
    axes[1, 0].set_ylabel('Размер окна λ')
    axes[1, 0].set_title('Динамика выбора оптимального размера окна')
    axes[1, 0].set_yticks([2, 3, 4, 5])
    axes[1, 0].grid(True, alpha=0.3)

    # График 4: Ошибки прогноза
    start_idx = min(var_model.possible_windows)
    axes[1, 1].bar(t[start_idx:], var_model.errors[start_idx:], alpha=0.7, color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].set_xlabel('Время')
    axes[1, 1].set_ylabel('Ошибка')
    axes[1, 1].set_title('Ошибки прогноза (переменное окно)')
    axes[1, 1].grid(True, alpha=0.3)

    # График 5: MSE для разных размеров окон (если есть история)
    if mse_history and len(mse_history) > 0:
        # Берем последние значения для наглядности
        display_points = min(20, len(mse_history))
        recent_history = mse_history[-display_points:]

        window_sizes = var_model.possible_windows
        mse_by_window = {ws: [] for ws in window_sizes}
        time_points = []

        for i, mse_dict in enumerate(recent_history, len(mse_history) - display_points + 1):
            time_points.append(i)
            for ws in window_sizes:
                mse_by_window[ws].append(mse_dict.get(ws, np.nan))

        for ws in window_sizes:
            if any(not np.isnan(v) for v in mse_by_window[ws]):
                axes[2, 0].plot(time_points, mse_by_window[ws], 'o-', label=f'λ={ws}',
                                markersize=4, linewidth=1.5)

        axes[2, 0].set_xlabel('Время')
        axes[2, 0].set_ylabel('MSE на скользящем окне')
        axes[2, 0].set_title('MSE для разных размеров окон')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

    # График 6: Сравнение MSE методов
    methods = ['Переменное окно', 'Фиксированное окно (λ=3)']
    mse_values = [var_model.calculate_metrics(data)['MSE'],
                  fixed_model.calculate_metrics(data)['MSE']]

    axes[2, 1].bar(methods, mse_values, alpha=0.7, color=['blue', 'green'])
    axes[2, 1].set_ylabel('MSE')
    axes[2, 1].set_title('Сравнение методов по MSE')
    axes[2, 1].grid(True, alpha=0.3, axis='y')

    # Добавление значений на столбцы
    for i, v in enumerate(mse_values):
        axes[2, 1].text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def main():
    """
    Основная функция для выполнения задания 4
    """
    # Исходные данные
    data = [50, 52, 54, 56, 58, 60, 75, 90, 92, 94, 96, 98, 100, 102, 104]

    print("Исходный временной ряд:")
    print(data)
    print(f"Длина ряда: {len(data)}")

    # Сравнение методов
    var_model, fixed_model, window_history = compare_with_fixed_window(data)

    # Анализ выбора окон
    print("\n4. АНАЛИЗ ВЫБОРА РАЗМЕРОВ ОКОН:")
    window_counts = {}
    for ws in [2, 3, 4, 5]:
        count = window_history.count(ws)
        window_counts[ws] = count
        percentage = count / len(window_history) * 100
        print(f"   λ={ws}: {count} раз ({percentage:.1f}%)")

    most_common = max(window_counts, key=window_counts.get)
    print(f"\n   Наиболее часто выбираемый размер окна: λ={most_common}")

    # Прогноз на следующий шаг
    print("\n5. ПРОГНОЗ НА СЛЕДУЮЩИЙ ШАГ:")
    var_next = var_model.forecast_next(data)
    fixed_next = fixed_model.forecast_next(data)

    print(f"   Переменное окно: {var_next:.2f}")
    print(f"   Фиксированное окно (λ=3): {fixed_next:.2f}")


if __name__ == "__main__":
    main()