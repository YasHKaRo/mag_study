from pr7_1 import ExponentialSmoothing
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# Используем класс ExponentialSmoothing из задания 1

def optimize_alpha(data: List[float], alphas: List[float]) -> Tuple[float, Dict[float, Dict[str, float]]]:
    """
    Поиск оптимального alpha для экспоненциального сглаживания

    Parameters:
    -----------
    data : List[float]
        Временной ряд
    alphas : List[float]
        Список значений alpha для перебора

    Returns:
    --------
    best_alpha : float
        Оптимальное значение alpha
    results : Dict[float, Dict[str, float]]
        Словарь с результатами для каждого alpha
    """
    results = {}
    best_alpha = None
    best_mse = float('inf')

    for alpha in alphas:
        try:
            model = ExponentialSmoothing(alpha=alpha)
            model.fit_predict(data)
            metrics = model.calculate_metrics(data)
            results[alpha] = metrics

            if metrics['MSE'] < best_mse:
                best_mse = metrics['MSE']
                best_alpha = alpha
        except:
            continue

    return best_alpha, results

def multi_step_forecast(model: ExponentialSmoothing, steps: int) -> List[float]:
    """
    Построение многошагового прогноза

    Parameters:
    -----------
    model : ExponentialSmoothing
        Обученная модель
    steps : int
        Число шагов прогноза

    Returns:
    --------
    List[float]
        Прогнозные значения
    """
    if not model.smoothed:
        raise ValueError("Модель не обучена")

    last_smoothed = model.smoothed[-1]
    return [last_smoothed] * steps


def plot_mse_vs_alpha(alphas: List[float], results: Dict[float, Dict[str, float]]):
    """
    Построение графика зависимости MSE от alpha
    """
    mse_values = [results[alpha]['MSE'] for alpha in alphas if alpha in results]

    plt.figure(figsize=(10, 6))
    plt.plot(alphas[:len(mse_values)], mse_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('α (параметр сглаживания)')
    plt.ylabel('MSE')
    plt.title('Зависимость MSE от параметра α')
    plt.grid(True, alpha=0.3)

    # Отметим оптимальное значение
    min_mse_idx = np.argmin(mse_values)
    plt.plot(alphas[min_mse_idx], mse_values[min_mse_idx], 'r*', markersize=15,
             label=f'Оптимальное α = {alphas[min_mse_idx]:.1f}\nMSE = {mse_values[min_mse_idx]:.4f}')
    plt.legend()
    plt.show()


def main():
    """
    Основная функция для выполнения задания 2
    """
    # Исходные данные
    data = [65.5, 66.2, 65.8, 67.1, 68.3, 67.9, 69.2, 70.5,
            69.8, 71.2, 72.5, 71.8, 73.1, 74.3, 73.6]

    print("=" * 80)
    print("ЗАДАНИЕ 2: ПРОГНОЗИРОВАНИЕ С ОПТИМИЗАЦИЕЙ ПАРАМЕТРОВ")
    print("=" * 80)

    # 1. Поиск оптимального alpha
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_alpha, results = optimize_alpha(data, alphas)

    print(f"\n1. РЕЗУЛЬТАТЫ ПЕРЕБОРА ПАРАМЕТРА α:")
    print(f"{'α':>6} | {'MSE':>12} | {'MAE':>12} | {'MAPE':>12}")
    print("-" * 60)

    for alpha in alphas:
        if alpha in results:
            print(f"{alpha:6.1f} | {results[alpha]['MSE']:12.4f} | "
                  f"{results[alpha]['MAE']:12.4f} | {results[alpha]['MAPE']:12.4f}")

    print(f"\n✓ Оптимальное значение α = {best_alpha:.1f}")
    print(f"✓ Минимальное MSE = {results[best_alpha]['MSE']:.4f}")

    # 2. Прогноз на 3 дня вперед с оптимальным alpha
    optimal_model = ExponentialSmoothing(alpha=best_alpha)
    optimal_model.fit_predict(data)
    forecast_steps = 3
    forecasts = multi_step_forecast(optimal_model, forecast_steps)

    print(f"\n2. ПРОГНОЗ НА 3 ДНЯ ВПЕРЕД (α = {best_alpha}):")
    for i, forecast in enumerate(forecasts, 1):
        print(f"  День {i}: {forecast:.2f}")

    # 3. Сравнение с наивной моделью (α = 1)
    naive_model = ExponentialSmoothing(alpha=1.0)  # α=1 соответствует наивной модели
    naive_model.fit_predict(data)
    naive_metrics = naive_model.calculate_metrics(data)

    print(f"\n3. СРАВНЕНИЕ С НАИВНОЙ МОДЕЛЬЮ (α = 1):")
    print(f"{'Метод':<25} | {'MSE':>12} | {'MAE':>12}")
    print("-" * 50)
    print(f"{f'Оптимальное (α={best_alpha})':<25} | "
          f"{results[best_alpha]['MSE']:12.4f} | {results[best_alpha]['MAE']:12.4f}")
    print(f"{'Наивная (α=1)':<25} | "
          f"{naive_metrics['MSE']:12.4f} | {naive_metrics['MAE']:12.4f}")

    improvement = (naive_metrics['MSE'] - results[best_alpha]['MSE']) / naive_metrics['MSE'] * 100
    print(f"\n✓ Улучшение MSE: {improvement:.1f}%")

    # 4. График зависимости MSE от alpha
    plot_mse_vs_alpha(alphas, results)

    # 5. Визуализация прогноза
    plt.figure(figsize=(12, 6))
    t_data = np.arange(len(data))
    t_forecast = np.arange(len(data), len(data) + forecast_steps)

    plt.plot(t_data, data, 'o-', label='Фактические данные', linewidth=2, markersize=6)
    plt.plot(t_data, optimal_model.forecasts, 's--', label='Прогноз на обучении',
             linewidth=1.5, markersize=4)
    plt.plot(t_forecast, forecasts, 'r*--', label='Прогноз на 3 дня вперед',
             linewidth=2, markersize=10)

    plt.xlabel('День')
    plt.ylabel('Курс валюты')
    plt.title(f'Прогнозирование курса валюты (оптимальное α={best_alpha})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()