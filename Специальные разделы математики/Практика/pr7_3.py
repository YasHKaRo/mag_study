import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox


class ARMAModel:
    """
    Класс для работы с моделью ARMA(1,1)
    """
    def __init__(self):
        """
        Инициализация модели
        """
        self.model = None
        self.results = None
        self.params = {}
        self.residuals = []
        self.forecasts = []
        self.forecast_conf_int = None

    # Исправленный метод fit в классе ARMAModel:
    def fit(self, data: List[float]):
        """
        Оценка параметров модели ARMA(1,1)

        Parameters:
        -----------
        data : List[float]
            Временной ряд
        """
        # Преобразуем в numpy array
        data_array = np.array(data)

        # Создаем и обучаем модель ARMA(1,1) = ARIMA(1,0,1)
        self.model = ARIMA(data_array, order=(1, 0, 1))
        self.results = self.model.fit()

        # Сохраняем параметры - исправленная версия
        self.params = {
            'a1': self.results.arparams[0] if len(self.results.arparams) > 0 else 0,
            'b1': self.results.maparams[0] if len(self.results.maparams) > 0 else 0,
            'const': self.results.params[0] if len(self.results.params) > 0 else 0
        }

        # Сохраняем остатки
        self.residuals = self.results.resid.tolist()

        # Получаем прогнозные значения на обучающей выборке
        self.forecasts = self.results.fittedvalues.tolist()
    def check_residuals(self, lags: int = 10) -> Dict:
        """
        Проверка остатков на некоррелированность

        Parameters:
        -----------
        lags : int
            Число лагов для теста

        Returns:
        --------
        dict
            Результаты теста Льюнга-Бокса
        """
        if len(self.residuals) == 0:
            raise ValueError("Модель не обучена")

        # Тест Льюнга-Бокса
        lb_test = acorr_ljungbox(self.residuals, lags=[lags], return_df=True)

        return {
            'test_statistic': lb_test['lb_stat'].iloc[0],
            'p_value': lb_test['lb_pvalue'].iloc[0],
            'is_white_noise': lb_test['lb_pvalue'].iloc[0] > 0.05
        }

    def forecast(self, steps: int = 4) -> Tuple[List[float], np.ndarray]:
        """
        Прогноз на steps шагов вперед

        Parameters:
        -----------
        steps : int
            Число шагов прогноза

        Returns:
        --------
        forecast_values : List[float]
            Точечный прогноз
        conf_int : np.ndarray
            Доверительный интервал
        """
        if self.results is None:
            raise ValueError("Модель не обучена")

        # Прогноз
        forecast_result = self.results.forecast(steps=steps)

        # Доверительный интервал (95%)
        forecast_object = self.results.get_forecast(steps=steps)
        conf_int = forecast_object.conf_int(alpha=0.05)

        return forecast_result.tolist(), conf_int
    def calculate_metrics(self, data: List[float]) -> Dict[str, float]:
        """
        Вычисление метрик качества модели

        Parameters:
        -----------
        data : List[float]
            Фактические значения

        Returns:
        --------
        dict
            Метрики качества
        """
        if not self.forecasts:
            raise ValueError("Модель не обучена")

        # Пропускаем первые значения (могут быть NaN из-за лагов)
        start_idx = 2  # Для ARMA(1,1) первые 2 значения могут быть NaN

        actual = np.array(data[start_idx:])
        pred = np.array(self.forecasts[start_idx:])
        errors = actual - pred

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


def main():
    """
    Основная функция для выполнения задания 3
    """
    # Исходные данные
    data = [100, 105, 108, 112, 115, 119, 122, 126, 129, 133, 136, 140]

    print("=" * 80)
    print("ЗАДАНИЕ 3: МОДЕЛЬ ARMA(1,1)")
    print("=" * 80)

    # 1. Построение и оценка модели
    print("\n1. ПОСТРОЕНИЕ МОДЕЛИ ARMA(1,1)")

    model = ARMAModel()
    model.fit(data)

    print(f"   Оцененные параметры:")
    print(f"     a₁ (параметр AR) = {model.params['a1']:.4f}")
    print(f"     b₁ (параметр MA) = {model.params['b1']:.4f}")
    print(f"     Константа = {model.params['const']:.4f}")

    # 2. Вычисление остатков
    print(f"\n2. ОСТАТКИ МОДЕЛИ:")
    print(f"   Первые 5 остатков: {model.residuals[:5]}")
    print(f"   Среднее остатков: {np.mean(model.residuals):.4f}")
    print(f"   Дисперсия остатков: {np.var(model.residuals):.4f}")

    # 3. Проверка адекватности модели
    print("\n3. ПРОВЕРКА АДЕКВАТНОСТИ МОДЕЛИ:")

    test_result = model.check_residuals(lags=5)
    print(f"   Тест Льюнга-Бокса:")
    print(f"     Статистика: {test_result['test_statistic']:.4f}")
    print(f"     p-значение: {test_result['p_value']:.4f}")

    if test_result['is_white_noise']:
        print(f"   ✓ Остатки являются белым шумом (p > 0.05)")
        print(f"   ✓ Модель адекватна")
    else:
        print(f"   ✗ Остатки не являются белым шумом (p ≤ 0.05)")
        print(f"   ✗ Модель неадекватна")

    # 4. Метрики качества
    metrics = model.calculate_metrics(data)
    print(f"\n4. МЕТРИКИ КАЧЕСТВА:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")

    # 5. Прогноз на 4 шага вперед
    print("\n5. ПРОГНОЗ НА 4 ШАГА ВПЕРЕД:")

    forecast_steps = 4
    forecast_values, conf_int = model.forecast(steps=forecast_steps)

    for i, (value, (lower, upper)) in enumerate(zip(forecast_values, conf_int), 1):
        print(f"   Шаг {i}: {value:.2f} (95% ДИ: [{lower:.2f}, {upper:.2f}])")

    # 6. Визуализация
    plt.figure(figsize=(14, 8))

    # Фактические данные и прогноз на обучении
    t_data = np.arange(len(data))
    t_forecast = np.arange(len(data), len(data) + forecast_steps)

    plt.subplot(2, 2, 1)
    plt.plot(t_data, data, 'o-', label='Фактические данные', linewidth=2, markersize=6)
    plt.plot(t_data[2:], model.forecasts[2:], 's--', label='Прогноз ARMA(1,1)',
             linewidth=1.5, markersize=4)
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.title('Модель ARMA(1,1): факт и прогноз')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Остатки
    plt.subplot(2, 2, 2)
    plt.plot(t_data[2:], model.residuals[2:], 'o-', color='red', linewidth=2, markersize=6)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Время')
    plt.ylabel('Остаток')
    plt.title('Остатки модели')
    plt.grid(True, alpha=0.3)

    # Гистограмма остатков
    plt.subplot(2, 2, 3)
    plt.hist(model.residuals[2:], bins=8, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Остаток')
    plt.ylabel('Частота')
    plt.title('Распределение остатков')
    plt.grid(True, alpha=0.3, axis='y')

    # Прогноз на будущие периоды
    plt.subplot(2, 2, 4)
    plt.plot(t_data, data, 'o-', label='История', linewidth=2, markersize=6)
    plt.plot(t_forecast, forecast_values, 'r*--', label='Прогноз',
             linewidth=2, markersize=10)
    plt.fill_between(t_forecast, conf_int[:, 0], conf_int[:, 1],
                     color='red', alpha=0.2, label='95% ДИ')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.title(f'Прогноз на {forecast_steps} шага вперед')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 7. Сводка модели
    print("\n" + "=" * 80)
    print("СВОДКА МОДЕЛИ ARMA(1,1):")
    print("=" * 80)
    print(model.results.summary())


if __name__ == "__main__":
    main()