import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse


class BayesianClassifier:
    """Байесовский классификатор для многомерных нормальных распределений"""

    def __init__(self):
        """
        Инициализация классификатора

        Атрибуты:
            classes (dict): Словарь для хранения параметров классов
            n_features (int or None): Количество признаков (определяется при добавлении первого класса)
        """
        self.classes = {}  # Словарь: имя_класса -> параметры_класса
        self.n_features = None  # Количество признаков будет определено позже

    def add_class(self, class_name: str, prior: float,
                  mean: np.ndarray, cov: np.ndarray):
        """
        Добавить класс в классификатор

        Параметры:
            class_name (str): Уникальное имя класса (например, 'A1', 'A2')
            prior (float): Априорная вероятность P(A_i) ∈ [0, 1]
            mean (np.ndarray): Вектор средних значений m_i размерности p
            cov (np.ndarray): Ковариационная матрица C_i размерности p×p

        Выбрасывает:
            ValueError: Если размерность признаков не согласована между классами
        """
        # Проверка согласованности размерностей
        if self.n_features is None:
            self.n_features = len(mean)  # Определяем размерность по первому классу
        elif len(mean) != self.n_features:
            raise ValueError("Размерность признаков должна быть одинаковой для всех классов")

        # Сохраняем параметры класса с предварительными вычислениями
        self.classes[class_name] = {
            'prior': prior,  # P(A_i)
            'mean': np.array(mean),  # m_i
            'cov': np.array(cov),  # C_i
            'cov_inv': np.linalg.inv(cov),  # C_i^(-1) - вычисляем один раз для производительности
            'cov_det': np.linalg.det(cov)  # |C_i| - вычисляем один раз
        }

    def discriminant_function(self, x: np.ndarray, class_name: str) -> float:
        """
        Вычислить значение дискриминантной функции для класса

        Формула: g_i(x) = ln(P(A_i)) - 0.5*ln(|C_i|) - 0.5*(x-m_i)^T * C_i^(-1) * (x-m_i)

        Параметры:
            x (np.ndarray): Вектор признаков размерности p
            class_name (str): Имя класса

        Возвращает:
            float: Значение дискриминантной функции g_i(x)
        """
        cls = self.classes[class_name]  # Получаем параметры класса
        diff = x - cls['mean']  # Разность (x - m_i)

        # Квадратичная форма Махаланобиса: (x-m_i)^T * C_i^(-1) * (x-m_i)
        # Используем оператор @ для матричного умножения
        mahalanobis_sq = diff.T @ cls['cov_inv'] @ diff

        # Вычисляем дискриминантную функцию
        g = (np.log(cls['prior']) -  # ln(P(A_i))
             0.5 * np.log(cls['cov_det']) -  # -0.5*ln(|C_i|)
             0.5 * mahalanobis_sq)  # -0.5*(расстояние Махаланобиса)^2

        return g

    def predict(self, x: np.ndarray):
        """
        Классифицировать образ (принять решение о принадлежности к классу)

        Решающее правило: x → argmax_i g_i(x)

        Параметры:
            x (np.ndarray): Вектор признаков

        Возвращает:
            tuple: (predicted_class, scores)
                predicted_class (str): Имя класса с максимальной g_i(x)
                scores (dict): Словарь {имя_класса: значение_g_i(x)} для всех классов
        """
        scores = {}  # Словарь для хранения значений дискриминантных функций

        # Вычисляем g_i(x) для каждого класса
        for class_name in self.classes:
            scores[class_name] = self.discriminant_function(x, class_name)

        # Находим класс с максимальным значением g_i(x)
        # key=scores.get означает: для каждого ключа получить значение из словаря scores
        predicted_class = max(scores, key=scores.get)

        return predicted_class, scores

    def predict_proba(self, x: np.ndarray):
        """
        Вычислить апостериорные вероятности классов по формуле Байеса

        Формула: P(A_i|x) = [P(A_i) * ω(x|A_i)] / Σ_j [P(A_j) * ω(x|A_j)]

        Параметры:
            x (np.ndarray): Вектор признаков

        Возвращает:
            dict: Словарь {имя_класса: апостериорная_вероятность}
        """
        likelihoods = {}  # P(A_i) * ω(x|A_i) для каждого класса

        # Вычисляем likelihood для каждого класса
        for class_name in self.classes:
            cls = self.classes[class_name]

            # Создаем объект многомерного нормального распределения
            rv = multivariate_normal(cls['mean'], cls['cov'])

            # Вычисляем ω(x|A_i) - плотность вероятности в точке x
            pdf_value = rv.pdf(x)

            # P(A_i) * ω(x|A_i)
            likelihoods[class_name] = cls['prior'] * pdf_value

        # Сумма likelihood по всем классам (нормировочная константа)
        total = sum(likelihoods.values())

        # Вычисляем апостериорные вероятности: делим каждый likelihood на сумму
        probas = {k: v / total for k, v in likelihoods.items()}

        return probas

    def mahalanobis_distance(self, class1: str, class2: str,
                             common_cov: np.ndarray = None) -> float:
        """
        Вычислить расстояние Махаланобиса между двумя классами

        Формула: r_ij = sqrt((m_i - m_j)^T * C^(-1) * (m_i - m_j))

        Параметры:
            class1, class2 (str): Имена классов
            common_cov (np.ndarray, optional): Общая ковариационная матрица.
                Если None, используется средняя матрица: (C1 + C2)/2

        Возвращает:
            float: Расстояние Махаланобиса
        """
        # Получаем векторы средних
        m1 = self.classes[class1]['mean']
        m2 = self.classes[class2]['mean']

        # Определяем общую ковариационную матрицу
        if common_cov is None:
            # Если не задана, используем среднюю из двух матриц
            c1 = self.classes[class1]['cov']
            c2 = self.classes[class2]['cov']
            common_cov = (c1 + c2) / 2

        # Вычисляем обратную матрицу
        cov_inv = np.linalg.inv(common_cov)

        # Разность векторов средних
        diff = m1 - m2

        # Вычисляем квадрат расстояния Махаланобиса
        r_sq = diff.T @ cov_inv @ diff

        # Возвращаем корень квадратный (собственно расстояние)
        return np.sqrt(r_sq)


def visualize_classifier_3classes(classifier: BayesianClassifier,
                                  x_range=(0, 15),
                                  y_range=(5, 25),
                                  test_points=None):
    """
    Визуализировать разделяющие границы для 3-х классов в 2D

    Параметры:
        classifier: Обученный байесовский классификатор
        x_range, y_range: Диапазоны значений по осям для визуализации
        test_points: Список тестовых точек для отображения на графике
    """
    # Проверяем, что классификатор работает с 2D данными
    if classifier.n_features != 2:
        raise ValueError("Визуализация только для 2D")

    # Создаем сетку точек для построения областей решений
    x1 = np.linspace(x_range[0], x_range[1], 300)  # Ось X
    x2 = np.linspace(y_range[0], y_range[1], 300)  # Ось Y
    X1, X2 = np.meshgrid(x1, x2)  # Создаем сетку 300x300 точек

    # Матрица для хранения меток классов
    Z = np.zeros_like(X1, dtype=int)

    # Получаем имена классов
    class_names = list(classifier.classes.keys())

    # Классифицируем каждую точку сетки (это самая медленная часть)
    print("Построение областей решений...")
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            # Создаем вектор признаков из координат точки
            x = np.array([X1[i, j], X2[i, j]])

            # Классифицируем точку
            pred_class, _ = classifier.predict(x)

            # Сохраняем индекс класса (0, 1 или 2)
            Z[i, j] = class_names.index(pred_class)

    # Создаем график
    plt.figure(figsize=(8, 6))

    # Цвета для фона областей классов
    colors_bg = ['lightblue', 'lightgreen', 'lightcoral']

    # Заливаем области разными цветами в зависимости от класса
    # levels=4 означает: границы между 0-1, 1-2, 2-3
    plt.contourf(X1, X2, Z, levels=len(class_names) + 1,
                 colors=colors_bg, alpha=0.5)

    # Рисуем границы между классами (черные линии)
    for i in range(1, len(class_names)):
        # Уровень i-0.5 соответствует границе между классами i-1 и i
        plt.contour(X1, X2, Z, levels=[i - 0.5],
                    colors='black', linewidths=2)

    # Цвета и маркеры для средних значений классов
    colors = ['blue', 'green', 'red']
    markers = ['o', '^', 's']  # Круг, треугольник, квадрат

    # Отображаем средние значения и эллипсы доверительных областей
    for idx, (class_name, cls) in enumerate(classifier.classes.items()):
        # Рисуем среднее значение класса
        plt.plot(cls['mean'][0], cls['mean'][1],
                 marker=markers[idx], markersize=15,
                 color=colors[idx], label=f'Среднее {class_name}',
                 markeredgecolor='black', markeredgewidth=2)

        # Вычисляем собственные значения и векторы ковариационной матрицы
        # Это нужно для построения эллипса, соответствующего форме распределения
        eigenvalues, eigenvectors = np.linalg.eig(cls['cov'])

        # Угол поворота эллипса определяется главным собственным вектором
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # Размеры эллипса (4σ соответствует ~95% доверительной области)
        width, height = 4 * np.sqrt(eigenvalues)  # 2σ в каждом направлении

        # Создаем эллипс
        ellipse = Ellipse(cls['mean'], width, height, angle=angle,
                          facecolor='none', edgecolor=colors[idx],
                          linewidth=2, linestyle='--')

        # Добавляем эллипс на график
        plt.gca().add_patch(ellipse)

    # Отображаем тестовые точки, если они заданы
    if test_points:
        for idx, point in enumerate(test_points):
            # Классифицируем тестовую точку
            pred_class, _ = classifier.predict(point)

            # Определяем цвет в соответствии с предсказанным классом
            color_idx = class_names.index(pred_class)

            # Рисуем тестовую точку (крестик большого размера)
            plt.plot(point[0], point[1], 'X', markersize=15,
                     color=colors[color_idx], markeredgecolor='black',
                     markeredgewidth=2,
                     # Добавляем в легенду только первые 2 точки для читаемости
                     label=f'Тест {idx + 1}: {pred_class}' if idx < 2 else None)

    # Настройки графика
    plt.xlabel('Признак x₁ (частота ключевых слов)')
    plt.ylabel('Признак x₂ (средняя длина предложения)')
    plt.title('Разделяющие границы байесовского классификатора (3 класса)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============ ОСНОВНАЯ ЧАСТЬ: ВЫПОЛНЕНИЕ ЗАДАНИЯ 2 ============

# Создаем экземпляр классификатора
classifier = BayesianClassifier()

# ------------ 1. Добавление классов (условие задачи) ------------
# Класс A1: технические документы
classifier.add_class('A1',
                     prior=0.5,  # P(A1) = 0.5
                     mean=np.array([8.0, 15.0]),  # m1 = [8.0, 15.0]^T
                     cov=np.array([[4.0, 1.0],  # C1 = [[4.0, 1.0],
                                   [1.0, 9.0]]))  # [1.0, 9.0]]

# Класс A2: научные статьи
classifier.add_class('A2',
                     prior=0.3,  # P(A2) = 0.3
                     mean=np.array([12.0, 20.0]),  # m2 = [12.0, 20.0]^T
                     cov=np.array([[6.0, 2.0],  # C2 = [[6.0, 2.0],
                                   [2.0, 16.0]]))  # [2.0, 16.0]]

# Класс A3: художественная литература
classifier.add_class('A3',
                     prior=0.2,  # P(A3) = 0.2
                     mean=np.array([5.0, 12.0]),  # m3 = [5.0, 12.0]^T
                     cov=np.array([[3.0, 0.5],  # C3 = [[3.0, 0.5],
                                   [0.5, 4.0]]))  # [0.5, 4.0]]

# ------------ 2. Тестовые точки из условия ------------
test_samples = [
    np.array([10.0, 18.0]),  # x⁽¹⁾
    np.array([6.0, 13.0])  # x⁽²⁾
]

# ------------ 2. Классификация документов ------------
print("=" * 60)
print("Классификация тестовых документов:")
print("=" * 60)

for i, x in enumerate(test_samples, 1):
    # Получаем предсказание и значения дискриминантных функций
    pred, scores = classifier.predict(x)

    print(f"\nДокумент {i}: x = {x}")
    print("-" * 40)

    # Выводим значения g_i(x) для каждого класса
    print("Значения дискриминантных функций:")
    for c, s in scores.items():
        print(f"  g_{c}(x) = {s:.4f}")

    # Выводим решение
    print(f"\nРешение: документ относится к классу → {pred}")

    # Вычисляем и выводим апостериорные вероятности
    probas = classifier.predict_proba(x)
    print("\nАпостериорные вероятности (по формуле Байеса):")
    for c, p in probas.items():
        print(f"  P({c}|x) = {p:.4f}  ({p * 100:.1f}%)")

# ------------ 3. Расстояния Махаланобиса ------------
print("\n" + "=" * 60)
print("Расстояния Махаланобиса между классами:")
print("=" * 60)

# Все пары классов
pairs = [('A1', 'A2'), ('A1', 'A3'), ('A2', 'A3')]

for c1, c2 in pairs:
    # Вычисляем расстояние Махаланобиса
    r = classifier.mahalanobis_distance(c1, c2)

    print(f"\nМежду {c1} и {c2}:")
    print(f"  Расстояние Махаланобиса = {r:.4f}")
    print(f"  Квадрат расстояния = {r ** 2:.4f}")

    # Интерпретация: чем больше расстояние, тем лучше разделимость классов
    if r < 3:
        print(f"  Интерпретация: слабая разделимость (< 3)")
    elif r < 6:
        print(f"  Интерпретация: умеренная разделимость (3-6)")
    else:
        print(f"  Интерпретация: хорошая разделимость (> 6)")

# ------------ 4. Визуализация ------------
print("\n" + "=" * 60)
print("Визуализация разделяющих границ...")
print("=" * 60)

# Запускаем визуализацию с тестовыми точками
visualize_classifier_3classes(classifier,
                              x_range=(0, 15),  # Диапазон по x1
                              y_range=(5, 25),  # Диапазон по x2
                              test_points=test_samples)  # Тестовые точки
