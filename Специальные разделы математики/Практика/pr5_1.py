import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, norm

Y = [20, 22, 19, 21, 20, 23, 22, 21, 20, 19]
# 1) Мат.ожидание, дисперсия и средневкадратичное по Y
y_mean = np.mean(Y)
variance_y = np.var(Y, ddof=1)
std_dev_y = np.std(Y, ddof=1)

print(f'Для Y: \n Мат.ожидание {y_mean}, \n Дисперсия {variance_y}, \n Среднеквадратичное {std_dev_y}')

Z = [5, 7, 8, 6, 5, 7, 8, 9, 6, 5]
# 2) Построение гистограммы по Z
plt.hist(Z, bins=5, edgecolor="red", alpha=0.7)
plt.title("Гистограмма")
plt.xlabel("Значение")
plt.ylim("Частота")
plt.show()

stat, p_value = shapiro(Z)
print(f'Статистика Шапиро-Уилка: {stat}, p_value: {p_value}')

# 3) Доверительный интервал по W
W = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]



