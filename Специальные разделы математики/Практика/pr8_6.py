import numpy as np
import matplotlib.pyplot as plt


class BayesianClassifier:
    """Байесовский классификатор для многомерных данных"""

    def __init__(self):
        """
        Инициализация классификатора

        Особенность задачи 6: 4-МЕРНОЕ ПРОСТРАНСТВО ПРИЗНАКОВ
        """
        self.classes = {}  # Словарь для параметров классов
        self.n_features = None  # Количество признаков

    def add_class(self, class_name: str, prior: float, mean: np.ndarray, cov: np.ndarray):
        """
        Добавить класс в классификатор

        Особенность: ДИАГОНАЛЬНЫЕ ковариационные матрицы
        Это означает, что признаки НЕЗАВИСИМЫ

        Параметры:
            class_name (str): Имя класса
            prior (float): Априорная вероятность P(A_i)
            mean (np.ndarray): Вектор средних m_i (4 элемента)
            cov (np.ndarray): Диагональная ковариационная матрица C_i (4×4)
        """
        if self.n_features is None:
            self.n_features = len(mean)  # Определяем размерность (4)
        elif len(mean) != self.n_features:
            raise ValueError("Размерность признаков должна быть одинаковой")

        self.classes[class_name] = {
            'prior': prior,
            'mean': np.array(mean),
            'cov': np.array(cov),
            'cov_inv': np.linalg.inv(cov),  # Для диагональной матрицы - просто обратные элементы
            'cov_det': np.linalg.det(cov)  # Для диагональной матрицы - произведение диагональных элементов
        }

    def discriminant_function(self, x: np.ndarray, class_name: str) -> float:
        """
        Вычислить значение дискриминантной функции

        Формула: g_i(x) = ln(P(A_i)) - 0.5*ln(|C_i|) - 0.5*(x-m_i)^T * C_i^(-1) * (x-m_i)

        УПРОЩЕНИЕ для диагональных матриц:
        - (x-m_i)^T * C_i^(-1) * (x-m_i) = Σ_j ((x_j - m_ij)^2 / σ_ij^2)
        - ln(|C_i|) = Σ_j ln(σ_ij^2)
        """
        cls = self.classes[class_name]
        diff = x - cls['mean']  # (x - m_i)

        # Квадратичная форма Махаланобиса
        # Для диагональной матрицы: sum((x_j - m_ij)^2 / σ_ij^2)
        mahalanobis_sq = diff.T @ cls['cov_inv'] @ diff

        # Дискриминантная функция
        g = (np.log(cls['prior']) -
             0.5 * np.log(cls['cov_det']) -
             0.5 * mahalanobis_sq)
        return g

    def predict(self, x: np.ndarray):
        """
        Классифицировать состояние сервера

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
        Вычислить расстояние Махаланобиса между классами

        Для диагональных матриц формула упрощается:
        r² = Σ_j ((m1_j - m2_j)^2 / σ_j^2)
        где σ_j^2 = (σ1_j^2 + σ2_j^2)/2
        """
        m1 = self.classes[class1]['mean']
        m2 = self.classes[class2]['mean']

        if common_cov is None:
            c1 = self.classes[class1]['cov']
            c2 = self.classes[class2]['cov']
            common_cov = (c1 + c2) / 2  # Средняя матрица

        cov_inv = np.linalg.inv(common_cov)  # Для диагональной - обратные элементы
        diff = m1 - m2  # Разность векторов средних

        r_sq = diff.T @ cov_inv @ diff  # Σ_j ((m1_j - m2_j)^2 / σ_j^2)
        return np.sqrt(r_sq)


# ============ ОСНОВНАЯ ЧАСТЬ: ВЫПОЛНЕНИЕ ЗАДАНИЯ 6 ============

print("=" * 80)
print("ЗАДАНИЕ 6: Многомерная классификация состояния сервера")
print("Мониторинг 4 параметров сервера в реальном времени")
print("=" * 80)

# ------------ ПАРАМЕТРЫ КЛАССОВ ------------
print("\nПАРАМЕТРЫ СИСТЕМЫ МОНИТОРИНГА:")

# Признаки (4 измерения):
# 1. x₁ - загрузка CPU (%)
# 2. x₂ - использование RAM (%)
# 3. x₃ - сетевой трафик (Мб/с)
# 4. x₄ - количество активных процессов

feature_names = ['x₁ (CPU %)', 'x₂ (RAM %)', 'x₃ (трафик Мб/с)', 'x₄ (процессы)']

# Класс A1: нормальная работа сервера
m1 = np.array([30, 50, 10, 100])
# Диагональная матрица: признаки НЕЗАВИСИМЫ
C1 = np.diag([100, 225, 25, 400])  # Дисперсии по каждому признаку
P1 = 0.9  # 90% времени сервер работает нормально

print("\nКласс A1 (нормальная работа):")
print(f"  Априорная вероятность: P(A1) = {P1} (90% времени)")
print(f"  Вектор средних m1 = {m1}")
print(f"    CPU: {m1[0]}%, RAM: {m1[1]}%, Трафик: {m1[2]} Мб/с, Процессы: {m1[3]}")
print(f"  Ковариационная матрица C1 (диагональная):")
print(f"    Дисперсии: CPU={C1[0, 0]}, RAM={C1[1, 1]}, Трафик={C1[2, 2]}, Процессы={C1[3, 3]}")
print(f"    Стандартные отклонения: CPU={np.sqrt(C1[0, 0]):.1f}, RAM={np.sqrt(C1[1, 1]):.1f}, " +
      f"Трафик={np.sqrt(C1[2, 2]):.1f}, Процессы={np.sqrt(C1[3, 3]):.1f}")

# Класс A2: перегрузка сервера
m2 = np.array([80, 90, 50, 200])
C2 = np.diag([225, 100, 100, 900])  # Дисперсии (большие - больше разброс при перегрузке)
P2 = 0.1  # 10% времени сервер перегружен

print("\nКласс A2 (перегрузка):")
print(f"  Априорная вероятность: P(A2) = {P2} (10% времени)")
print(f"  Вектор средних m2 = {m2}")
print(f"    CPU: {m2[0]}%, RAM: {m2[1]}%, Трафик: {m2[2]} Мб/с, Процессы: {m2[3]}")
print(f"  Ковариационная матрица C2 (диагональная):")
print(f"    Дисперсии: CPU={C2[0, 0]}, RAM={C2[1, 1]}, Трафик={C2[2, 2]}, Процессы={C2[3, 3]}")
print(f"    Стандартные отклонения: CPU={np.sqrt(C2[0, 0]):.1f}, RAM={np.sqrt(C2[1, 1]):.1f}, " +
      f"Трафик={np.sqrt(C2[2, 2]):.1f}, Процессы={np.sqrt(C2[3, 3]):.1f}")

# ------------ 1. ПОСТРОЕНИЕ 4-МЕРНОГО КЛАССИФИКАТОРА ------------
print("\n" + "=" * 80)
print("1. ПОСТРОЕНИЕ 4-МЕРНОГО КЛАССИФИКАТОРА")
print("=" * 80)

classifier = BayesianClassifier()
classifier.add_class('A1', P1, m1, C1)
classifier.add_class('A2', P2, m2, C2)

print(f"\nКлассификатор создан:")
print(f"  Размерность пространства: {classifier.n_features} признака")
print(f"  Количество классов: {len(classifier.classes)}")
print(f"  Тип ковариационных матриц: ДИАГОНАЛЬНЫЕ (признаки независимы)")
print(f"\nВАЖНО: Диагональные матрицы означают, что:")
print(f"  1. Признаки НЕ коррелируют между собой")
print(f"  2. Квадратичная форма Махаланобиса упрощается:")
print(f"     (x-m)^T * C^(-1) * (x-m) = Σ ((x_i - m_i)^2 / σ_i^2)")
print(f"  3. Это евклидово расстояние с весами, обратными дисперсиям")

# ------------ 2. КЛАССИФИКАЦИЯ СОСТОЯНИЙ СЕРВЕРА ------------
print("\n" + "=" * 80)
print("2. КЛАССИФИКАЦИЯ СОСТОЯНИЙ СЕРВЕРА")
print("=" * 80)

# Тестовые состояния сервера
test_samples = [
    np.array([50, 70, 25, 150]),  # Состояние 1: умеренная нагрузка
    np.array([75, 85, 45, 180])  # Состояние 2: высокая нагрузка
]

print("\nТестовые состояния сервера:")

for i, x in enumerate(test_samples, 1):
    print(f"\nСостояние {i}:")
    print(f"  CPU: {x[0]}%, RAM: {x[1]}%, Трафик: {x[2]} Мб/с, Процессы: {x[3]}")
    print(f"  Отклонение от нормального режима (m1):")
    print(f"    CPU: {x[0] - m1[0]:+d}%, RAM: {x[1] - m1[1]:+d}%, " +
          f"Трафик: {x[2] - m1[2]:+d} Мб/с, Процессы: {x[3] - m1[3]:+d}")
    print(f"  Отклонение от режима перегрузки (m2):")
    print(f"    CPU: {x[0] - m2[0]:+d}%, RAM: {x[1] - m2[1]:+d}%, " +
          f"Трафик: {x[2] - m2[2]:+d} Мб/с, Процессы: {x[3] - m2[3]:+d}")

print("\nРезультаты классификации:")

for i, x in enumerate(test_samples, 1):
    # Получаем предсказание
    pred, scores = classifier.predict(x)

    print(f"\nСостояние {i}: x = {x}")
    print("-" * 60)

    # Значения дискриминантных функций
    g1 = scores['A1']
    g2 = scores['A2']

    print(f"  g_A1(x) = {g1:.4f}")
    print(f"  g_A2(x) = {g2:.4f}")

    # Разность и решение
    diff = g1 - g2
    print(f"\n  g_A1(x) - g_A2(x) = {diff:.4f}")

    if diff > 0:
        print(f"  Решение: → A1 (нормальная работа)")
        print(f"  Преимущество нормального режима: {diff:.4f}")
    else:
        print(f"  Решение: → A2 (перегрузка)")
        print(f"  Преимущество режима перегрузки: {-diff:.4f}")

    # Детальный анализ по признакам
    print(f"\n  Анализ по признакам (вклад в разность g1-g2):")
    print(f"  {'Признак':<20} | {'(x-m1)²/σ1²':>12} | {'(x-m2)²/σ2²':>12} | {'Вклад':>12}")
    print(f"  {'-' * 20} | {'-' * 12} | {'-' * 12} | {'-' * 12}")

    total_contrib = 0
    for j in range(4):
        # Вклад j-го признака в квадратичные формы
        contrib1 = (x[j] - m1[j]) ** 2 / C1[j, j]
        contrib2 = (x[j] - m2[j]) ** 2 / C2[j, j]
        contrib_diff = contrib2 - contrib1  # Вклад в разность g1-g2 (с обратным знаком)
        total_contrib += 0.5 * contrib_diff  # 0.5 из формулы g_i(x)

        print(f"  {feature_names[j]:<20} | {contrib1:12.4f} | {contrib2:12.4f} | {contrib_diff:12.4f}")

    print(f"  {'Суммарный вклад':<20} | {'':12} | {'':12} | {total_contrib:12.4f}")

# ------------ 3. РАССТОЯНИЕ МАХАЛАНОБИСА МЕЖДУ КЛАССАМИ ------------
print("\n" + "=" * 80)
print("3. РАССТОЯНИЕ МАХАЛАНОБИСА")
print("=" * 80)

# Средняя ковариационная матрица
C_avg = (C1 + C2) / 2
print(f"\nСредняя ковариационная матрица C_avg:")
print(f"  Дисперсии: CPU={C_avg[0, 0]:.1f}, RAM={C_avg[1, 1]:.1f}, " +
      f"Трафик={C_avg[2, 2]:.1f}, Процессы={C_avg[3, 3]:.1f}")

# Вычисляем расстояние Махаланобиса
r = classifier.mahalanobis_distance('A1', 'A2', C_avg)

print(f"\nРасстояние Махаланобиса: r = {r:.4f}")
print(f"Квадрат расстояния: r² = {r ** 2:.4f}")

print(f"\nРасчет по формуле для диагональных матриц:")
print(f"  r² = Σ_j ((m1_j - m2_j)² / σ²_avg_j)")
print(f"  где σ²_avg_j = (σ1_j² + σ2_j²)/2")

for j in range(4):
    diff = m1[j] - m2[j]
    sigma_avg = (C1[j, j] + C2[j, j]) / 2
    contrib = diff ** 2 / sigma_avg
    print(f"  {feature_names[j]:<20}: ({diff:6.1f})² / {sigma_avg:6.1f} = {contrib:6.4f}")

print(f"  {'Сумма':<20}: {'':26} = {r ** 2:6.4f}")

print(f"\nИнтерпретация расстояния r = {r:.2f}:")
if r < 3:
    print("  Слабая разделимость (классы плохо разделены)")
elif r < 6:
    print("  Умеренная разделимость")
elif r < 10:
    print("  Хорошая разделимость")
else:
    print("  Отличная разделимость (классы хорошо разделены)")

# Вероятность ошибки (приближенная)
print(f"\nПриближенная вероятность ошибки классификации:")
p_error_approx = np.exp(-0.5 * r ** 2) / (r * np.sqrt(2 * np.pi))
print(f"  P_ош ≈ exp(-r²/2) / (r√(2π)) = {p_error_approx:.6f} ({p_error_approx * 100:.4f}%)")

# ------------ 4. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ------------
print("\n" + "=" * 80)
print("4. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ (FEATURE IMPORTANCE)")
print("=" * 80)

print("\nМетрики важности признаков:")
print(f"{'Признак':<25} | {'|m1-m2|':>10} | {'σ_avg':>10} | {'r_i = |m1-m2|/σ':>12} | {'Вклад в r² (%)':>12}")
print(f"{'-' * 25} | {'-' * 10} | {'-' * 10} | {'-' * 12} | {'-' * 12}")

total_r_sq = r ** 2
feature_contributions = []

for i in range(4):
    # Абсолютная разность средних
    dm = abs(m1[i] - m2[i])

    # Среднее стандартное отклонение
    sigma_avg = np.sqrt((C1[i, i] + C2[i, i]) / 2)

    # Относительное расстояние по одному признаку
    r_i = dm / sigma_avg

    # Вклад в общее расстояние Махаланобиса
    contrib = (dm ** 2) / (sigma_avg ** 2)
    contrib_percent = (contrib / total_r_sq) * 100

    feature_contributions.append((feature_names[i], contrib_percent))

    print(f"{feature_names[i]:<25} | {dm:10.1f} | {sigma_avg:10.1f} | {r_i:12.4f} | {contrib_percent:12.2f}%")

print(f"\nСумма вкладов: {sum(c[1] for c in feature_contributions):.2f}%")

# Сортируем по важности
feature_contributions_sorted = sorted(feature_contributions, key=lambda x: x[1], reverse=True)

print(f"\nРанжирование признаков по важности:")
for rank, (name, percent) in enumerate(feature_contributions_sorted, 1):
    print(f"  {rank}. {name}: {percent:.1f}% вклада в разделимость")

print(f"\nИнтерпретация:")
print("1. Самый важный признак вносит наибольший вклад в расстояние Махаланобиса")
print("2. Для улучшения классификации можно:")
print("   - Более точно измерять важные признаки")
print("   - Добавить пороги по важным признакам для быстрой классификации")
print("   - Уменьшать шум (дисперсию) важных признаков")

# ------------ 5. ВИЗУАЛИЗАЦИЯ ПРОЕКЦИЙ НА 2D ПОДПРОСТРАНСТВА ------------
print("\n" + "=" * 80)
print("5. ПРОЕКЦИИ НА 2D ПОДПРОСТРАНСТВА")
print("=" * 80)

print(f"\nТак как пространство 4-мерное, невозможно визуализировать его целиком.")
print("Строим проекции на наиболее информативные пары признаков...")

# Выбираем пары для визуализации (первые 3 пары с самыми важными признаками)
pairs = [(0, 1), (0, 2), (0, 3)]  # CPU с RAM, Traffic, Processes

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

print(f"\nПроекции на плоскости:")
for ax_idx, (i, j) in enumerate(pairs):
    ax = axes[ax_idx]

    print(f"\n{feature_names[i]} - {feature_names[j]}:")
    print(f"  Класс A1: ({m1[i]:.1f}, {m1[j]:.1f})")
    print(f"  Класс A2: ({m2[i]:.1f}, {m2[j]:.1f})")

    # Определяем границы для визуализации
    x_min = min(m1[i] - 3 * np.sqrt(C1[i, i]), m2[i] - 3 * np.sqrt(C2[i, i]))
    x_max = max(m1[i] + 3 * np.sqrt(C1[i, i]), m2[i] + 3 * np.sqrt(C2[i, i]))
    y_min = min(m1[j] - 3 * np.sqrt(C1[j, j]), m2[j] - 3 * np.sqrt(C2[j, j]))
    y_max = max(m1[j] + 3 * np.sqrt(C1[j, j]), m2[j] + 3 * np.sqrt(C2[j, j]))

    # Создаем сетку в 2D
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # Для классификации в каждой точке сетки фиксируем остальные признаки на средних значениях
    mean_other = (m1 + m2) / 2  # Среднее значение по всем признакам

    Z = np.zeros_like(xx)  # Матрица для меток классов

    print(f"  Классификация точек сетки...")
    for p in range(xx.shape[0]):
        for q in range(xx.shape[1]):
            # Создаем полный 4-мерный вектор:
            # - по осям i, j берем значения из сетки
            # - по остальным осям - средние значения
            x_full = mean_other.copy()
            x_full[i] = xx[p, q]  # Значение по первой оси проекции
            x_full[j] = yy[p, q]  # Значение по второй оси проекции

            # Классифицируем
            pred, _ = classifier.predict(x_full)
            Z[p, q] = 0 if pred == 'A1' else 1

    # Визуализация областей решений
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                colors=['lightblue', 'lightcoral'], alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    # Отображаем средние значения классов
    ax.plot(m1[i], m1[j], 'o', markersize=10, color='blue',
            label='Среднее A1 (норма)')
    ax.plot(m2[i], m2[j], 's', markersize=10, color='red',
            label='Среднее A2 (перегрузка)')

    # Отображаем тестовые точки
    for k, x in enumerate(test_samples, 1):
        pred, _ = classifier.predict(x)
        color = 'blue' if pred == 'A1' else 'red'
        marker = 'X'
        ax.plot(x[i], x[j], marker, markersize=12, color=color,
                label=f'Тест {k}: {pred}' if ax_idx == 0 else None)

    # Настройки графика
    ax.set_xlabel(feature_names[i])
    ax.set_ylabel(feature_names[j])
    ax.set_title(f'Проекция: {feature_names[i]} - {feature_names[j]}')
    ax.grid(True, alpha=0.3)

    if ax_idx == 0:
        ax.legend()

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("ЗАДАНИЕ 6 ВЫПОЛНЕНО УСПЕШНО!")
print("=" * 80)

print("\nКлючевые выводы по задаче 6:")
print("1. Система мониторинга успешно классифицирует состояния сервера по 4 признакам")
print("2. Диагональные ковариационные матрицы упрощают вычисления (признаки независимы)")
print(f"3. Расстояние Махаланобиса r = {r:.2f} показывает {'хорошую' if r > 3 else 'слабую'} разделимость")
print("4. Наиболее важный признак для обнаружения перегрузки: " +
      f"{feature_contributions_sorted[0][0]} ({feature_contributions_sorted[0][1]:.1f}% вклада)")
print("5. Визуализация проекций помогает понять структуру данных в многомерном пространстве")
print("6. Для реальной системы можно рекомендовать:")
print("   - Мониторить в первую очередь важные признаки")
print("   - Настроить пороги срабатывания по важным признакам")
print("   - Регулярно переоценивать параметры на новых данных")