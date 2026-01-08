import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf  # Функция ошибок для вычисления вероятности ошибки
from matplotlib.patches import Ellipse  # Для рисования эллипсов доверительных областей


class BayesianClassifier:
    """Байесовский классификатор для задач с нормальным распределением признаков"""

    def __init__(self):
        """
        Инициализация классификатора

        Атрибуты:
            classes (dict): Словарь для хранения параметров классов
            n_features (int or None): Количество признаков (определяется при добавлении первого класса)
        """
        self.classes = {}  # Словарь: имя_класса -> параметры_класса
        self.n_features = None  # Количество признаков

    def add_class(self, class_name: str, prior: float,
                  mean: np.ndarray, cov: np.ndarray):
        """
        Добавить класс в классификатор

        Особенность задачи 3: ОБЩАЯ КОВАРИАЦИОННАЯ МАТРИЦА для обоих классов

        Параметры:
            class_name (str): Уникальное имя класса
            prior (float): Априорная вероятность P(A_i)
            mean (np.ndarray): Вектор средних значений m_i
            cov (np.ndarray): Ковариационная матрица C_i
        """
        # Проверка согласованности размерностей
        if self.n_features is None:
            self.n_features = len(mean)
        elif len(mean) != self.n_features:
            raise ValueError("Размерность должна быть одинаковой")

        # Сохраняем параметры класса с предварительными вычислениями
        self.classes[class_name] = {
            'prior': prior,
            'mean': np.array(mean),
            'cov': np.array(cov),
            'cov_inv': np.linalg.inv(cov),  # C_i^(-1)
            'cov_det': np.linalg.det(cov)  # |C_i|
        }

    def discriminant_function(self, x: np.ndarray, class_name: str) -> float:
        """
        Вычислить значение дискриминантной функции для класса

        Формула: g_i(x) = ln(P(A_i)) - 0.5*ln(|C_i|) - 0.5*(x-m_i)^T * C_i^(-1) * (x-m_i)

        Особенность: При равных матрицах C_1 = C_2 = C, функция становится ЛИНЕЙНОЙ
        """
        cls = self.classes[class_name]
        diff = x - cls['mean']  # (x - m_i)

        # Квадратичная форма Махаланобиса
        mahalanobis_sq = diff.T @ cls['cov_inv'] @ diff

        # Дискриминантная функция
        g = (np.log(cls['prior']) -
             0.5 * np.log(cls['cov_det']) -
             0.5 * mahalanobis_sq)
        return g

    def predict(self, x: np.ndarray):
        """
        Классифицировать образ

        Решающее правило: x → argmax_i g_i(x)
        """
        scores = {}
        for class_name in self.classes:
            scores[class_name] = self.discriminant_function(x, class_name)
        predicted_class = max(scores, key=scores.get)
        return predicted_class, scores

    def mahalanobis_distance(self, class1: str, class2: str,
                             common_cov: np.ndarray = None) -> float:
        """
        Вычислить расстояние Махаланобиса между двумя классами

        Особенность задачи 3: Используется ОБЩАЯ ковариационная матрица

        Формула: r = sqrt((m1 - m2)^T * C^(-1) * (m1 - m2))

        Параметры:
            class1, class2 (str): Имена классов
            common_cov (np.ndarray): Общая ковариационная матрица

        Возвращает:
            float: Расстояние Махаланобиса
        """
        m1 = self.classes[class1]['mean']
        m2 = self.classes[class2]['mean']

        # В задаче 3 всегда используется общая матрица
        if common_cov is None:
            common_cov = self.classes[class1]['cov']

        cov_inv = np.linalg.inv(common_cov)  # Обратная матрица
        diff = m1 - m2  # Разность векторов средних

        # Квадрат расстояния Махаланобиса
        r_sq = diff.T @ cov_inv @ diff

        return np.sqrt(r_sq)


def visualize_linear_classifier(classifier: BayesianClassifier,
                                x_range=(100, 220),
                                y_range=(50, 170)):
    """
    Визуализировать линейную разделяющую границу

    Особенность: При равных ковариационных матрицах граница - ПРЯМАЯ ЛИНИЯ
    """
    # Создаем сетку точек
    x1 = np.linspace(x_range[0], x_range[1], 300)
    x2 = np.linspace(y_range[0], y_range[1], 300)
    X1, X2 = np.meshgrid(x1, x2)

    # Матрица для меток классов
    Z = np.zeros_like(X1, dtype=int)
    class_names = list(classifier.classes.keys())

    # Классифицируем каждую точку сетки
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i, j], X2[i, j]])
            pred_class, _ = classifier.predict(x)
            Z[i, j] = class_names.index(pred_class)

    # Создаем график
    plt.figure(figsize=(8, 6))

    # Заливаем области разными цветами
    colors_bg = ['lightblue', 'lightcoral']
    plt.contourf(X1, X2, Z, colors=colors_bg, alpha=0.5)

    # Рисуем разделяющую границу (уровень 0.5 соответствует границе между классами 0 и 1)
    plt.contour(X1, X2, Z, levels=[0.5], colors='black', linewidths=3)

    # Отображаем средние значения классов
    colors = ['blue', 'red']
    markers = ['o', 's']  # Круг и квадрат
    for idx, (class_name, cls) in enumerate(classifier.classes.items()):
        plt.plot(cls['mean'][0], cls['mean'][1],
                 marker=markers[idx], markersize=15,
                 color=colors[idx], label=f'Среднее {class_name}',
                 markeredgecolor='black', markeredgewidth=2)

        # Рисуем эллипс доверительной области (2σ = ~95%)
        eigenvalues, eigenvectors = np.linalg.eig(cls['cov'])

        # Угол поворота эллипса
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # Размеры эллипса (4σ в каждом направлении)
        width, height = 4 * np.sqrt(eigenvalues)  # 2σ

        # Создаем и добавляем эллипс
        ellipse = Ellipse(cls['mean'], width, height, angle=angle,
                          facecolor='none', edgecolor=colors[idx],
                          linewidth=2, linestyle='--')
        plt.gca().add_patch(ellipse)

    # Настройки графика
    plt.xlabel('x₁ — время нажатия клавиши (мс)')
    plt.ylabel('x₂ — время между нажатиями (мс)')
    plt.title('Линейная разделяющая граница байесовского классификатора\n(равные ковариационные матрицы)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============ ОСНОВНАЯ ЧАСТЬ: ВЫПОЛНЕНИЕ ЗАДАНИЯ 3 ============

print("=" * 70)
print("ЗАДАНИЕ 3: Классификация с равными ковариационными матрицами")
print("Система биометрической аутентификации по времени нажатия клавиш")
print("=" * 70)

# ------------ ОБЩАЯ КОВАРИАЦИОННАЯ МАТРИЦА ------------
# Задана в условии задачи: C = [[400, 50], [50, 225]]
C_common = np.array([[400, 50], [50, 225]])
print("\nОбщая ковариационная матрица C (для обоих пользователей):")
print(C_common)
print(f"Определитель |C| = {np.linalg.det(C_common):.0f}")

# ------------ СОЗДАНИЕ И НАСТРОЙКА КЛАССИФИКАТОРА ------------
classifier = BayesianClassifier()

# Добавляем Пользователя 1
classifier.add_class(
    'Пользователь 1',
    prior=0.5,  # P(A1) = 0.5
    mean=np.array([150.0, 100.0]),  # m1 = [150, 100]^T
    cov=C_common  # Общая матрица
)

# Добавляем Пользователя 2
classifier.add_class(
    'Пользователь 2',
    prior=0.5,  # P(A2) = 0.5
    mean=np.array([180.0, 120.0]),  # m2 = [180, 120]^T
    cov=C_common  # Та же общая матрица
)

print("\nПараметры классов:")
print("Пользователь 1: m1 = [150.0, 100.0]^T, P(A1) = 0.5")
print("Пользователь 2: m2 = [180.0, 120.0]^T, P(A2) = 0.5")
print("\nВАЖНО: Одинаковая ковариационная матрица C для обоих классов!")
print("Это обеспечивает ЛИНЕЙНУЮ разделяющую границу.")

# ------------ 1. УРАВНЕНИЕ РАЗДЕЛЯЮЩЕЙ ГИПЕРПЛОСКОСТИ ------------
print("\n" + "=" * 70)
print("1. Уравнение разделяющей гиперплоскости:")
print("=" * 70)

# Вычисляем обратную матрицу
C_inv = np.linalg.inv(C_common)
print(f"\nОбратная матрица C^(-1):")
print(C_inv)

# Векторы средних
m1 = np.array([150.0, 100.0])
m2 = np.array([180.0, 120.0])
print(f"\nРазность векторов средних: m1 - m2 = {m1 - m2}")

# Коэффициенты разделяющей гиперплоскости
# w = C^(-1) * (m1 - m2)
w = C_inv @ (m1 - m2)

# Свободный член w0 = -0.5 * (m1 + m2)^T * C^(-1) * (m1 - m2)
# Упрощенная форма при равных априорных вероятностях
w0 = -0.5 * (m1 + m2) @ w

print(f"\nКоэффициенты разделяющей гиперплоскости:")
print(f"w = {w}")  # Вектор нормали к гиперплоскости
print(f"w₀ = {w0:.4f}")  # Свободный член

# Уравнение гиперплоскости: w^T * x + w0 = 0
print(f"\nУравнение разделяющей прямой:")
print(f"g(x) = {w[0]:.4f}·x₁ + {w[1]:.4f}·x₂ + {w0:.4f} = 0")

print("\nИнтерпретация:")
print("- Если g(x) > 0 → принадлежит Пользователю 1")
print("- Если g(x) < 0 → принадлежит Пользователю 2")
print("- Если g(x) = 0 → точка на границе принятия решения")

# Явный вид уравнения прямой (если нужно выразить x₂ через x₁)
if abs(w[1]) > 1e-10:  # Избегаем деления на ноль
    k = -w[0] / w[1]  # Угловой коэффициент
    b = -w0 / w[1]  # Смещение
    print(f"\nУравнение в явном виде (x₂ через x₁):")
    print(f"x₂ = {k:.4f}·x₁ + {b:.4f}")

# ------------ 2. ЛИНЕЙНЫЙ КЛАССИФИКАТОР ПОСТРОЕН ------------
print("\n" + "=" * 70)
print("2. Линейный классификатор построен в классе BayesianClassifier")
print("=" * 70)
print("Классификатор использует общую ковариационную матрицу,")
print("что гарантирует линейность разделяющей границы.")

# ------------ 3. РАССТОЯНИЕ МАХАЛАНОБИСА ------------
print("\n" + "=" * 70)
print("3. Расстояние Махаланобиса между классами:")
print("=" * 70)

# Вычисляем расстояние Махаланобиса
r = classifier.mahalanobis_distance('Пользователь 1', 'Пользователь 2', C_common)

print(f"\nРасстояние Махаланобиса: r = {r:.4f}")
print(f"Квадрат расстояния: r² = {r ** 2:.4f}")

# Геометрическая интерпретация
print("\nГеометрическая интерпретация:")
print(f"- r = {r:.2f} - расстояние между центрами классов в единицах")
print("  стандартного отклонения с учетом корреляции признаков")
print("- Чем больше r, тем лучше разделимость классов")
print(f"- В данной задаче r = {r:.2f} указывает на {'умеренную' if r < 4 else 'хорошую'} разделимость")

# ------------ 4. ВЕРОЯТНОСТЬ ОШИБКИ КЛАССИФИКАЦИИ ------------
print("\n" + "=" * 70)
print("4. Оценка вероятности ошибки классификации:")
print("=" * 70)

# Вероятность ошибки для двух классов с нормальным распределением
# и равными ковариационными матрицами:
# P_ош = 0.5 * (1 - erf(r/(2√2)))
p_error = 0.5 * (1 - erf(r / (2 * np.sqrt(2))))

print(f"\nРасчет вероятности ошибки:")
print(f"P_ош = 0.5 * (1 - erf(r/(2√2)))")
print(f"     = 0.5 * (1 - erf({r:.4f}/(2√2)))")
print(f"     = 0.5 * (1 - erf({r / (2 * np.sqrt(2)):.4f}))")
print(f"     = {p_error:.4f} ({p_error * 100:.2f}%)")

print("\nИнтерпретация:")
print(f"- Система будет ошибаться примерно в {p_error * 100:.1f}% случаев")
print(f"- Это соответствует 1 ошибке на каждые {int(1 / p_error)} попыток")
print("- Для биометрической аутентификации это {'приемлемый' if p_error < 0.05 else 'высокий'} уровень ошибки")

# Альтернативная формула через функцию Лапласа
print(f"\nАльтернативный расчет через функцию Лапласа Φ:")
print(f"P_ош = Φ(-r/2) = Φ(-{r / 2:.4f}) ≈ {p_error:.4f}")

# ------------ 5. ВИЗУАЛИЗАЦИЯ ЛИНЕЙНОЙ РАЗДЕЛЯЮЩЕЙ ГРАНИЦЫ ------------
print("\n" + "=" * 70)
print("5. Визуализация линейной разделяющей границы")
print("=" * 70)
print("Строится график с областями решений, средними значениями,")
print("эллипсами доверительных областей и разделяющей прямой...")

# Вызываем функцию визуализации
visualize_linear_classifier(classifier)

print("\n" + "=" * 70)
print("ОБЪЯСНЕНИЕ ГРАФИКА:")
print("=" * 70)
print("1. СИНИЕ ОБЛАСТИ → Пользователь 1")
print("2. КРАСНЫЕ ОБЛАСТИ → Пользователь 2")
print("3. ЧЕРНАЯ ЛИНИЯ → разделяющая граница (гиперплоскость)")
print("4. КРУГ и КВАДРАТ → средние значения классов")
print("5. ПУНКТИРНЫЕ ЭЛЛИПСЫ → области 2σ (≈95% доверительные интервалы)")
print("6. ЭЛЛИПСЫ ОДИНАКОВОЙ ФОРМЫ → следствие общей ковариационной матрицы")
print("7. ГРАНИЦА - ПРЯМАЯ ЛИНИЯ → следствие равных ковариационных матриц")

print("\n" + "=" * 70)
print("ЗАДАНИЕ 3 ВЫПОЛНЕНО УСПЕШНО!")
print("=" * 70)
print("\nВыводы по задаче 3:")
print("1. При равных ковариационных матрицах разделяющая граница ЛИНЕЙНА")
print("2. Уравнение границы: w^T·x + w₀ = 0")
print(f"3. Расстояние Махаланобиса r = {r:.2f}")
print(f"4. Вероятность ошибки P_ош = {p_error * 100:.2f}%")
print("5. Система может использоваться для биометрической аутентификации")
print("   с умеренной точностью.")