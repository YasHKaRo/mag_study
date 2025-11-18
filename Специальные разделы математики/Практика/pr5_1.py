import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, norm, t

Y = [20, 22, 19, 21, 20, 23, 22, 21, 20, 19]
# 1) Мат.ожидание, дисперсия и средневкадратичное по Y
y_mean = np.mean(Y)
variance_y = np.var(Y, ddof=1)
std_dev_y = np.std(Y, ddof=1)

print(f'Для Y: \n Мат.ожидание {y_mean}, \n Дисперсия {variance_y}, \n Среднеквадратичное {std_dev_y}')

Z = [5, 7, 8, 6, 5, 7, 8, 9, 6, 5]
# 2) Построение гистограммы по Z
plt.hist(Z, bins=5, edgecolor='black', alpha=0.7)
plt.title("Гистограмма распределения для выборки Z")
plt.xlabel("Значения")
plt.ylabel("Частота")
plt.grid(alpha=0.3)
plt.show()

stat, p_value = shapiro(Z)
print(f'Статистика Шапиро-Уилка: {stat}, p_value: {p_value}')

# 3) Доверительный интервал по W
W = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
confidence = 0.99
n_W = len(W)
mean_W = np.mean(W)
std_dev_W = np.std(W, ddof=1)

# Для малой выборки используем t-распределение
t_critical = t.ppf((1 + confidence) / 2, n_W - 1)
margin_of_error = t_critical * (std_dev_W / np.sqrt(n_W))
lower_bound = mean_W - margin_of_error
upper_bound = mean_W + margin_of_error

print(f"Доверительный интервал для математического ожидания: [{round(lower_bound, 2)}, {round(upper_bound, 2)}]")

# 4) Корреляционный момент и коэффициент корреляции.

A = [10, 12, 14, 16, 18, 20]
B = [5, 6, 7, 8, 9, 10]

# Корреляционный момент (ковариация)
covariance = np.cov(A, B, ddof=1)[0, 1]
# Коэффициент корреляции Пирсона
correlation = np.corrcoef(A, B)[0, 1]

print(f"Корреляционный момент (ковариация): {covariance}")
print(f"Коэффициент корреляции: {correlation}")

# 5) Объем выборки, необходимый для оценки математического
# ожидания с погрешностью 0.5 и доверительной вероятностью 95%.
V = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
E = 0.5  # Погрешность
confidence_5 = 0.95

# Оцениваем стандартное отклонение по выборке V
sigma_est = np.std(V, ddof=1)

# Находим z-критическое значение для 95% доверительной вероятности
z_critical = norm.ppf((1 + confidence_5) / 2)

# Вычисляем необходимый объем выборки
n_required = (z_critical * sigma_est / E) ** 2

print(f"Оценка стандартного отклонения по выборке V: {sigma_est}")
print(f"Критическое значение z: {z_critical}")
print(f"Необходимый объем выборки: {n_required}")

