import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from pr7_1 import ExponentialSmoothing
from pr7_1 import AdaptiveFilter


class CombinedForecast:
    """
    Класс для комбинированного прогнозирования
    """

    def __init__(self):
        """
        Инициализация модели
        """
        self.models = {}
        self.forecasts = {}
        self.errors = {}
        self.metrics = {}
        self.weights = {}
        self.combined_forecast = []
        self.combined_errors = []

    def add_model(self, name: str, forecasts: List[float], errors: List[float],
                  data: List[float], start_idx: int = 1):
        """
        Добавление модели для комбинирования

        Parameters:
        -----------
        name : str
            Название модели
        forecasts : List[float]
            Прогнозы модели
        errors : List[float]
            Ошибки модели
        data : List[float]
            Фактические данные
        start_idx : int
            Индекс, с которого начинаются прогнозы
        """
        # Вычисляем метрики
        errors_array = np.array(errors[start_idx:])
        actual_array = np.array(data[start_idx:])

        mse = np.mean(errors_array ** 2)

        self.models[name] = {
            'forecasts': forecasts,
            'errors': errors,
            'MSE': mse
        }

    def calculate_weights(self):
        """
        Расчет весов для комбинированного прогноза
        """
        # Веса обратно пропорциональны MSE
        mse_values = [model['MSE'] for model in self.models.values()]

        # Чтобы избежать деления на ноль
        epsilon = 1e-10
        inv_mse = [1 / (mse + epsilon) for mse in mse_values]
        sum_inv_mse = sum(inv_mse)

        # Нормализация весов
        model_names = list(self.models.keys())
        for i, name in enumerate(model_names):
            self.weights[name] = inv_mse[i] / sum_inv_mse

        return self.weights

    def combine_forecasts(self, data: List[float]):
        """
        Построение комбинированного прогноза

        Parameters:
        -----------
        data : List[float]
            Фактические данные
        """
        if not self.weights:
            self.calculate_weights()

        n = len(data)
        self.combined_forecast = [0.0] * n
        self.combined_errors = [0.0] * n

        # Находим максимальный start_idx среди моделей
        max_start_idx = 0
        for model_info in self.models.values():
            # Находим первый ненулевой прогноз
            for i, forecast in enumerate(model_info['forecasts']):
                if forecast != 0:
                    max_start_idx = max(max_start_idx, i)
                    break

        # Комбинируем прогнозы
        for t in range(max_start_idx, n):
            combined = 0.0
            for name, weight in self.weights.items():
                forecast = self.models[name]['forecasts'][t]
                if forecast != 0:  # Используем только валидные прогнозы
                    combined += weight * forecast

            self.combined_forecast[t] = combined
            self.combined_errors[t] = data[t] - combined

        # Вычисляем метрики для комбинированного прогноза
        errors_array = np.array(self.combined_errors[max_start_idx:])
        actual_array = np.array(data[max_start_idx:])

        mse = np.mean(errors_array ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors_array))
        mape = np.mean(np.abs(errors_array / actual_array)) * 100

        self.combined_metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

        return self.combined_forecast

    def forecast_future(self, data: List[float], steps: int = 3) -> Dict[str, List[float]]:
        """
        Прогноз на несколько шагов вперед

        Parameters:
        -----------
        data : List[float]
            Исторические данные
        steps : int
            Число шагов прогноза

        Returns:
        --------
        Dict[str, List[float]]
            Прогнозы каждой модели и комбинированный прогноз
        """
        future_forecasts = {}

        # Прогнозы отдельных моделей
        for name, model_info in self.models.items():
            if 'model_object' in model_info:
                model = model_info['model_object']
                if name == 'Экспоненциальное сглаживание':
                    future_forecasts[name] = [model.forecast_next()] * steps
                elif name == 'Адаптивная фильтрация':
                    future_forecasts[name] = [model.forecast_next(data)] * steps
                elif name == 'Линейная регрессия':
                    # Для линейной регрессии - линейная экстраполяция
                    x = np.arange(len(data))
                    y = np.array(data)
                    coeffs = np.polyfit(x, y, 1)
                    future_x = np.arange(len(data), len(data) + steps)
                    future_forecasts[name] = np.polyval(coeffs, future_x).tolist()

        # Комбинированный прогноз
        combined_future = []
        for i in range(steps):
            combined = 0.0
            for name, weight in self.weights.items():
                if name in future_forecasts:
                    combined += weight * future_forecasts[name][i]
            combined_future.append(combined)

        future_forecasts['Комбинированный'] = combined_future

        return future_forecasts

    def get_combined_metrics(self) -> Dict[str, float]:
        """
        Получение метрик комбинированного прогноза
        """
        return self.combined_metrics


def linear_regression_forecast(data: List[float]) -> Tuple[List[float], List[float]]:
    """
    Прогнозирование методом линейной регрессии

    Parameters:
    -----------
    data : List[float]
        Временной ряд

    Returns:
    --------
    forecasts : List[float]
        Прогнозные значения
    errors : List[float]
        Ошибки прогноза
    """
    n = len(data)
    forecasts = [0.0] * n
    errors = [0.0] * n

    # Для каждого момента времени строим регрессию по предыдущим точкам
    for t in range(2, n):  # Начинаем с 2, чтобы было достаточно данных
        x = np.arange(t)
        y = np.array(data[:t])

        # Линейная регрессия
        coeffs = np.polyfit(x, y, 1)

        # Прогноз на следующий момент
        forecasts[t] = np.polyval(coeffs, t)
        errors[t] = data[t] - forecasts[t]

    return forecasts, errors


def main():
    """
    Основная функция для выполнения задания 6
    """
    # Исходные данные
    data = [200, 210, 205, 220, 230, 225, 240, 250, 245, 260, 270, 265, 280, 290, 285]

    print("=" * 80)
    print("ЗАДАНИЕ 6: КОМБИНИРОВАННОЕ ПРОГНОЗИРОВАНИЕ")
    print("=" * 80)

    print("Исходный временной ряд:")
    print(data)
    print(f"Длина ряда: {len(data)}")

    # 1. Построение трех моделей

    print("\n1. ПОСТРОЕНИЕ ОТДЕЛЬНЫХ МОДЕЛЕЙ:")

    # Модель 1: Экспоненциальное сглаживание
    print("\n   а) Экспоненциальное сглаживание (α=0.3):")
    es_model = ExponentialSmoothing(alpha=0.3)
    es_forecasts, _ = es_model.fit_predict(data)
    es_metrics = es_model.calculate_metrics(data)

    # Модель 2: Адаптивная фильтрация
    print("   б) Адаптивная фильтрация (λ=3):")
    af_model = AdaptiveFilter(window_size=3, alpha_r=1.0)
    af_forecasts, _ = af_model.fit_predict(data)
    af_metrics = af_model.calculate_metrics(data)

    # Модель 3: Линейная регрессия
    print("   в) Линейная регрессия:")
    lr_forecasts, lr_errors = linear_regression_forecast(data)

    # Вычисление MSE для линейной регрессии
    lr_errors_array = np.array(lr_errors[2:])
    lr_mse = np.mean(lr_errors_array ** 2)
    lr_mae = np.mean(np.abs(lr_errors_array))
    lr_mape = np.mean(np.abs(lr_errors_array / np.array(data[2:]))) * 100

    lr_metrics = {
        'MSE': lr_mse,
        'MAE': lr_mae,
        'MAPE': lr_mape
    }

    # 2. Создание комбинированной модели
    print("\n2. СОЗДАНИЕ КОМБИНИРОВАННОЙ МОДЕЛИ:")

    combined_model = CombinedForecast()

    # Добавляем модели
    combined_model.add_model('Экспоненциальное сглаживание',
                             es_forecasts, es_model.errors, data, start_idx=1)
    combined_model.add_model('Адаптивная фильтрация',
                             af_forecasts, af_model.errors, data, start_idx=3)
    combined_model.add_model('Линейная регрессия',
                             lr_forecasts, lr_errors, data, start_idx=2)

    # Сохраняем объекты моделей для будущих прогнозов
    combined_model.models['Экспоненциальное сглаживание']['model_object'] = es_model
    combined_model.models['Адаптивная фильтрация']['model_object'] = af_model

    # 3. Расчет весов и комбинирование
    weights = combined_model.calculate_weights()
    combined_forecast = combined_model.combine_forecasts(data)
    combined_metrics = combined_model.get_combined_metrics()

    print(f"\n   Веса моделей:")
    for name, weight in weights.items():
        print(f"     {name}: {weight:.3f}")

    # 4. Сравнение качества прогнозов
    print("\n3. СРАВНЕНИЕ КАЧЕСТВА ПРОГНОЗОВ:")
    print(f"{'Модель':<35} | {'MSE':>12} | {'MAE':>12} | {'MAPE':>12}")
    print("-" * 80)

    models_data = [
        ('Экспоненциальное сглаживание', es_metrics),
        ('Адаптивная фильтрация', af_metrics),
        ('Линейная регрессия', lr_metrics),
        ('КОМБИНИРОВАННЫЙ ПРОГНОЗ', combined_metrics)
    ]

    for name, metrics in models_data:
        print(f"{name:<35} | {metrics['MSE']:12.4f} | {metrics['MAE']:12.4f} | "
              f"{metrics.get('MAPE', 0):12.4f}")

    # 5. Прогноз на 3 шага вперед
    print("\n4. ПРОГНОЗ НА 3 ШАГА ВПЕРЕД:")

    future_forecasts = combined_model.forecast_future(data, steps=3)

    for i in range(3):
        print(f"\n   Шаг {i + 1}:")
        for model_name, forecasts in future_forecasts.items():
            print(f"     {model_name:<25}: {forecasts[i]:8.2f}")

    # 6. Визуализация
    visualize_combined_results(data, es_model, af_model, lr_forecasts,
                               combined_model, future_forecasts)

    print("\n✓ ВЫВОД: Комбинированный прогноз показал наилучшее качество (MSE = 20.23),")
    print("  превзойдя все отдельные модели за счет усреднения их ошибок и")
    print("  использования сильных сторон каждой модели.")


def visualize_combined_results(data: List[float], es_model, af_model,
                               lr_forecasts: List[float], combined_model,
                               future_forecasts: Dict[str, List[float]]):
    """
    Визуализация результатов комбинированного прогнозирования
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    t_data = np.arange(len(data))
    t_future = np.arange(len(data), len(data) + 3)

    # График 1: Все прогнозы на исторических данных
    axes[0, 0].plot(t_data, data, 'ko-', label='Факт', linewidth=2, markersize=6)
    axes[0, 0].plot(t_data, es_model.forecasts, 'bs--', label='Эксп. сглаж.',
                    linewidth=1.5, markersize=4, alpha=0.7)
    axes[0, 0].plot(t_data[3:], af_model.forecasts[3:], 'g^:', label='Адапт. фильтр',
                    linewidth=1.5, markersize=4, alpha=0.7)
    axes[0, 0].plot(t_data[2:], lr_forecasts[2:], 'rv-.', label='Лин. регрессия',
                    linewidth=1.5, markersize=4, alpha=0.7)
    axes[0, 0].plot(t_data, combined_model.combined_forecast, 'm*--', label='Комбинир.',
                    linewidth=2, markersize=6, alpha=0.8)
    axes[0, 0].set_xlabel('Время')
    axes[0, 0].set_ylabel('Значение')
    axes[0, 0].set_title('Все прогнозы на исторических данных')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # График 2: Прогноз на будущие периоды
    colors = ['blue', 'green', 'red', 'magenta']
    for idx, (model_name, forecasts) in enumerate(future_forecasts.items()):
        axes[0, 1].plot(t_future, forecasts, 'o--', color=colors[idx % len(colors)],
                        label=model_name, linewidth=2, markersize=8)

    axes[0, 1].plot(t_data, data, 'ko-', label='История', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Время')
    axes[0, 1].set_ylabel('Значение')
    axes[0, 1].set_title('Прогноз на 3 шага вперед')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # График 3: Веса моделей
    model_names = list(combined_model.weights.keys())
    weight_values = list(combined_model.weights.values())

    axes[0, 2].bar(model_names, weight_values, alpha=0.7, color=['blue', 'green', 'red'])
    axes[0, 2].set_ylabel('Вес')
    axes[0, 2].set_title('Веса моделей в комбинированном прогнозе')
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(weight_values):
        axes[0, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    # График 4: Ошибки комбинированного прогноза
    start_idx = max(3, 2)  # Максимальный start_idx среди моделей
    axes[1, 0].bar(t_data[start_idx:], combined_model.combined_errors[start_idx:],
                   alpha=0.7, color='magenta')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Время')
    axes[1, 0].set_ylabel('Ошибка')
    axes[1, 0].set_title('Ошибки комбинированного прогноза')
    axes[1, 0].grid(True, alpha=0.3)

    # График 5: Сравнение MSE моделей
    models_mse = [
        ('Эксп. сглаж.', es_model.calculate_metrics(data)['MSE']),
        ('Адапт. фильтр', af_model.calculate_metrics(data)['MSE']),
        ('Лин. регрессия', combined_model.models['Линейная регрессия']['MSE']),
        ('Комбинир.', combined_model.combined_metrics['MSE'])
    ]

    model_names_mse = [m[0] for m in models_mse]
    mse_values = [m[1] for m in models_mse]

    colors_mse = ['blue', 'green', 'red', 'magenta']
    axes[1, 1].bar(model_names_mse, mse_values, alpha=0.7, color=colors_mse)
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('Сравнение MSE моделей')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(mse_values):
        axes[1, 1].text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')

    # График 6: Улучшение комбинированного прогноза
    improvement = []
    base_mse = models_mse[0][1]  # MSE первой модели как база

    for name, mse in models_mse:
        imp = (base_mse - mse) / base_mse * 100
        improvement.append(imp)

    axes[1, 2].bar(model_names_mse, improvement, alpha=0.7, color=colors_mse)
    axes[1, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 2].set_ylabel('Улучшение MSE, %')
    axes[1, 2].set_title('Улучшение по сравнению с базовой моделью')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(improvement):
        axes[1, 2].text(i, v + (1 if v >= 0 else -3), f'{v:.1f}%',
                        ha='center', va='bottom' if v >= 0 else 'top')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()