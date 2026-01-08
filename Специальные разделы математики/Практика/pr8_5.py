import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import auc  # Для вычисления площади под ROC-кривой


class BayesianClassifier:
    """Байесовский классификатор для обнаружения спама"""

    def __init__(self):
        """
        Инициализация классификатора

        Особенность задачи 5: СИЛЬНЫЙ ДИСБАЛАНС КЛАССОВ
        P(A1) = 0.95 (легитимная почта)
        P(A2) = 0.05 (спам)
        """
        self.classes = {}  # Словарь для параметров классов

    def add_class(self, class_name: str, prior: float, mean: np.ndarray, cov: np.ndarray):
        """
        Добавить класс в классификатор

        Параметры:
            class_name (str): Имя класса
            prior (float): Априорная вероятность P(A_i)
            mean (np.ndarray): Вектор средних m_i
            cov (np.ndarray): Ковариационная матрица C_i
        """
        self.classes[class_name] = {
            'prior': prior,  # P(A_i)
            'mean': mean,  # m_i
            'cov': cov,  # C_i
            'cov_inv': np.linalg.inv(cov),  # C_i^(-1)
            'cov_det': np.linalg.det(cov)  # |C_i|
        }

    def discriminant_function(self, x: np.ndarray, class_name: str) -> float:
        """
        Вычислить дискриминантную функцию

        Формула: g_i(x) = ln(P(A_i)) - 0.5*ln(|C_i|) - 0.5*(x-m_i)^T * C_i^(-1) * (x-m_i)
        """
        cls = self.classes[class_name]
        diff = x - cls['mean']
        mahalanobis_sq = diff.T @ cls['cov_inv'] @ diff
        g = np.log(cls['prior']) - 0.5 * np.log(cls['cov_det']) - 0.5 * mahalanobis_sq
        return g

    def predict(self, x: np.ndarray):
        """
        Классифицировать письмо

        Решающее правило: x → argmax_i g_i(x)
        """
        scores = {name: self.discriminant_function(x, name) for name in self.classes}
        predicted_class = max(scores, key=scores.get)
        return predicted_class, scores


# ============ ОСНОВНАЯ ЧАСТЬ: ВЫПОЛНЕНИЕ ЗАДАНИЯ 5 ============

print("=" * 70)
print("ЗАДАНИЕ 5: Классификация с неравными априорными вероятностями")
print("Обнаружение спама в электронной почте")
print("=" * 70)

# ------------ ПАРАМЕТРЫ КЛАССОВ ИЗ УСЛОВИЯ ------------
print("\nПАРАМЕТРЫ КЛАССОВ:")

# Класс A1: Легитимная почта
m1 = np.array([0.5, 1.0])  # Мало восклицательных знаков, мало ссылок
C1 = np.array([[0.25, 0.1],
               [0.1, 0.5]])  # Малый разброс признаков

print("\nA1 (легитимная почта):")
print(f"  Вектор средних m1 = [{m1[0]}, {m1[1]}]ᵀ")
print(f"  Ковариационная матрица C1 =")
print(f"  [[{C1[0, 0]}, {C1[0, 1]}],")
print(f"   [{C1[1, 0]}, {C1[1, 1]}]]")
print("  Интерпретация: обычные письма имеют мало восклицательных знаков (0.5) и ссылок (1.0)")

# Класс A2: Спам
m2 = np.array([3.0, 5.0])  # Много восклицательных знаков, много ссылок
C2 = np.array([[1.0, 0.5],
               [0.5, 2.0]])  # Больший разброс признаков

print("\nA2 (спам):")
print(f"  Вектор средних m2 = [{m2[0]}, {m2[1]}]ᵀ")
print(f"  Ковариационная матрица C2 =")
print(f"  [[{C2[0, 0]}, {C2[0, 1]}],")
print(f"   [{C2[1, 0]}, {C2[1, 1]}]]")
print("  Интерпретация: спам содержит много восклицательных знаков (3.0) и ссылок (5.0)")
print("  Большие дисперсии: спам более разнообразен по форме")

# ------------ 1. ПОСТРОЕНИЕ КЛАССИФИКАТОРА ------------
print("\n" + "=" * 70)
print("1. ПОСТРОЕНИЕ КЛАССИФИКАТОРА С ИСХОДНЫМИ АПРИОРНЫМИ ВЕРОЯТНОСТЯМИ")
print("=" * 70)

# Создаем классификатор
classifier = BayesianClassifier()

# Добавляем классы с заданными априорными вероятностями
classifier.add_class('A1 (легитимная)', 0.95, m1, C1)  # 95% писем - легитимные
classifier.add_class('A2 (спам)', 0.05, m2, C2)  # 5% писем - спам

print(f"\nАприорные вероятности:")
print(f"P(A₁) = 0.95 (легитимная почта)")
print(f"P(A₂) = 0.05 (спам)")
print(f"Отношение P(A₂)/P(A₁) = {0.05 / 0.95:.4f}")

print(f"\nВАЖНО: Сильный дисбаланс классов!")
print("Это означает, что классификатор будет 'осторожен' в отнесении писем к спаму.")
print("Даже если признаки указывают на спам, априорная информация (95% легитимных)")
print("будет тянуть решение в сторону класса A1.")

# ------------ 2. КЛАССИФИКАЦИЯ ТЕСТОВЫХ ПИСЕМ ------------
print("\n" + "=" * 70)
print("2. КЛАССИФИКАЦИЯ ПИСЕМ")
print("=" * 70)

# Тестовые письма из условия
test_samples = [
    np.array([1.5, 2.5]),  # Письмо 1: умеренное количество признаков
    np.array([2.5, 4.0])  # Письмо 2: много признаков (ближе к спаму)
]

print("\nТестовые письма:")
for i, x in enumerate(test_samples, 1):
    print(f"\nПисьмо {i}: x = [{x[0]}, {x[1]}]ᵀ")
    print(f"  x₁ = {x[0]} восклицательных знаков")
    print(f"  x₂ = {x[1]} ссылок")
    print(f"  Расстояние до m1: {np.linalg.norm(x - m1):.2f}")
    print(f"  Расстояние до m2: {np.linalg.norm(x - m2):.2f}")

print("\nРезультаты классификации:")

for i, x in enumerate(test_samples, 1):
    # Получаем предсказание
    pred, scores = classifier.predict(x)

    print(f"\nПисьмо {i}: x = [{x[0]}, {x[1]}]ᵀ")
    print("-" * 40)

    # Значения дискриминантных функций
    g1 = scores['A1 (легитимная)']
    g2 = scores['A2 (спам)']

    print(f"  g_A1(x) = {g1:.4f}")
    print(f"  g_A2(x) = {g2:.4f}")

    # Решение
    print(f"\n  Решение: {pred}")

    # Объяснение решения
    if pred == 'A1 (легитимная)':
        print(f"  Причина: g_A1(x) > g_A2(x) на {g1 - g2:.4f}")
        print(f"  Априорная информация (P(A1)=0.95) перевесила признаки")
    else:
        print(f"  Причина: g_A2(x) > g_A1(x) на {g2 - g1:.4f}")
        print(f"  Признаки спама достаточно сильные, чтобы перевесить априорную информацию")

# ------------ 3. ИССЛЕДОВАНИЕ ВЛИЯНИЯ АПРИОРНЫХ ВЕРОЯТНОСТЕЙ ------------
print("\n" + "=" * 70)
print("3. ИССЛЕДОВАНИЕ ВЛИЯНИЯ АПРИОРНЫХ ВЕРОЯТНОСТЕЙ")
print("=" * 70)

# Разные комбинации априорных вероятностей
prior_pairs = [
    (0.95, 0.05),  # Исходные: 95% легитимных, 5% спама
    (0.80, 0.20),  # 80% легитимных, 20% спама
    (0.50, 0.50),  # Равные вероятности
    (0.20, 0.80)  # 20% легитимных, 80% спама (обратный дисбаланс)
]

print("\nКлассификация при разных априорных вероятностях:")
print(f"{'P(A₁)':>8} | {'P(A₂)':>8} | {'Порог θ=P(A₂)/P(A₁)':>25} | {'Письмо 1':>15} | {'Письмо 2':>15}")
print("-" * 80)

for p1, p2 in prior_pairs:
    # Создаем временный классификатор с текущими вероятностями
    clf_temp = BayesianClassifier()
    clf_temp.add_class('A1', p1, m1, C1)
    clf_temp.add_class('A2', p2, m2, C2)

    # Классифицируем тестовые письма
    pred1, _ = clf_temp.predict(test_samples[0])
    pred2, _ = clf_temp.predict(test_samples[1])

    # Вычисляем порог
    threshold = p2 / p1

    print(f"{p1:8.2f} | {p2:8.2f} | {threshold:25.4f} | {pred1:>15} | {pred2:>15}")

print("\nИнтерпретация:")
print("1. При P(A1)=0.95: классификатор очень 'осторожен', редко помечает как спам")
print("2. При P(A1)=0.50: равные вероятности, решение принимается только по признакам")
print("3. При P(A1)=0.20: классификатор 'агрессивен', часто помечает как спам")
print("4. Порог θ показывает, насколько убедительными должны быть признаки спама")
print("   чтобы перевесить априорную информацию")

# ------------ 4. и 5. ROC-КРИВАЯ И ОПТИМАЛЬНЫЙ ПОРОГ ------------
print("\n" + "=" * 70)
print("4. и 5. ROC-КРИВАЯ И ОПТИМАЛЬНЫЙ ПОРОГ")
print("=" * 70)

print("\nГенерация искусственной выборки для анализа ROC...")
print(f"Всего писем: {n_samples}")
print(f"Легитимных: {int(n_samples * 0.95)} (95%)")
print(f"Спама: {int(n_samples * 0.05)} (5%)")

# Фиксируем seed для воспроизводимости
np.random.seed(42)

# Генерируем искусственные данные согласно заданным распределениям
n_samples = 1000
data_A1 = np.random.multivariate_normal(m1, C1, int(n_samples * 0.95))
data_A2 = np.random.multivariate_normal(m2, C2, int(n_samples * 0.05))

# Объединяем данные
data = np.vstack((data_A1, data_A2))
labels = np.array([0] * len(data_A1) + [1] * len(data_A2))  # 0=легитимное, 1=спам


def likelihood_ratio_ln(x):
    """
    Вычислить логарифм отношения правдоподобия

    Формула: ln(ω(x|A2) / ω(x|A1))

    Большие положительные значения: признаки сильнее указывают на спам
    Отрицательные значения: признаки сильнее указывают на легитимное письмо
    """
    pdf1 = multivariate_normal.pdf(x, mean=m1, cov=C1)  # ω(x|A1)
    pdf2 = multivariate_normal.pdf(x, mean=m2, cov=C2)  # ω(x|A2)
    return np.log(pdf2 / pdf1 + 1e-12)  # +1e-12 для избежания деления на ноль


print("\nВычисление отношения правдоподобия для каждого письма...")
scores = np.array([likelihood_ratio_ln(x) for x in data])

print(f"\nСтатистика по ln(L_spam/L_legit):")
print(f"  Минимум: {scores.min():.4f} (явно легитимное)")
print(f"  Максимум: {scores.max():.4f} (явно спам)")
print(f"  Среднее: {scores.mean():.4f}")
print(f"  Медиана: {np.median(scores):.4f}")

# Варьируем порог принятия решения
print("\nВарьирование порога и вычисление TPR/FPR...")
thresholds = np.linspace(scores.min() - 2, scores.max() + 2, 500)

tpr_list = []  # True Positive Rate (чувствительность)
fpr_list = []  # False Positive Rate (ложные срабатывания)

for th in thresholds:
    # Предсказываем спам, если score > порога
    pred_spam = scores > th

    # Вычисляем матрицу ошибок
    tp = np.sum(pred_spam & (labels == 1))  # True Positive: спам правильно обнаружен
    fp = np.sum(pred_spam & (labels == 0))  # False Positive: легитимное помечено как спам
    tn = np.sum(~pred_spam & (labels == 0))  # True Negative: легитимное правильно распознано
    fn = np.sum(~pred_spam & (labels == 1))  # False Negative: спам пропущен

    # Вычисляем метрики
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR = TP / (TP + FN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # FPR = FP / (FP + TN)

    tpr_list.append(tpr)
    fpr_list.append(fpr)

# Вычисляем площадь под ROC-кривой (AUC)
roc_auc = auc(fpr_list, tpr_list)

print(f"\nПлощадь под ROC-кривой (AUC) = {roc_auc:.4f}")
if roc_auc > 0.9:
    print("Отличная разделимость классов!")
elif roc_auc > 0.8:
    print("Хорошая разделимость классов.")
elif roc_auc > 0.7:
    print("Удовлетворительная разделимость.")
else:
    print("Слабая разделимость классов.")

# Находим оптимальный порог по индексу Юдена
print("\nПоиск оптимального порога по индексу Юдена (J = TPR - FPR)...")
youden_j = np.array(tpr_list) - np.array(fpr_list)  # Индекс Юдена
idx_opt = np.argmax(youden_j)  # Индекс максимального значения J
opt_th = thresholds[idx_opt]  # Оптимальный порог
opt_tpr = tpr_list[idx_opt]  # TPR при оптимальном пороге
opt_fpr = fpr_list[idx_opt]  # FPR при оптимальном пороге

print(f"\nОПТИМАЛЬНЫЙ ПОРОГ:")
print(f"  ln(L_spam / L_legit) > {opt_th:.4f} → классифицировать как спам")
print(f"\nМетрики при оптимальном пороге:")
print(f"  TPR (True Positive Rate) = {opt_tpr:.4f}")
print(f"    → Обнаружено {opt_tpr * 100:.1f}% спама")
print(f"  FPR (False Positive Rate) = {opt_fpr:.4f}")
print(f"    → {opt_fpr * 100:.1f}% легитимных писем ошибочно помечены как спам")
print(f"  FNR (False Negative Rate) = {1 - opt_tpr:.4f}")
print(f"    → Пропущено {(1 - opt_tpr) * 100:.1f}% спама")
print(f"  Точность (Precision) = {opt_tpr / (opt_tpr + opt_fpr):.4f}")
print(f"    → {opt_tpr / (opt_tpr + opt_fpr) * 100:.1f}% писем, помеченных как спам, действительно спам")

# Сравнение с исходным классификатором (P(A1)=0.95, P(A2)=0.05)
print(f"\nСравнение с исходным классификатором (P(A1)=0.95):")
print("Исходный порог θ = P(A2)/P(A1) = 0.05/0.95 = 0.0526")
print(f"Исходный ln(θ) = ln(0.0526) = {np.log(0.05 / 0.95):.4f}")
print(f"Оптимальный порог = {opt_th:.4f}")
print(f"Разница: оптимальный порог {'выше' if opt_th > np.log(0.05 / 0.95) else 'ниже'} исходного")

if opt_th > np.log(0.05 / 0.95):
    print("→ Для минимизации ложных срабатываний нужен БОЛЕЕ СТРОГИЙ критерий спама")
    print("→ Меньше писем будет помечено как спам, но и меньше ложных срабатываний")
else:
    print("→ Можно использовать БОЛЕЕ ЛОЯЛЬНЫЙ критерий спама")

# ------------ ВИЗУАЛИЗАЦИЯ ROC-КРИВОЙ ------------
print("\n" + "=" * 70)
print("ВИЗУАЛИЗАЦИЯ ROC-КРИВОЙ")
print("=" * 70)

plt.figure(figsize=(10, 8))

# 1. ROC-кривая
plt.subplot(2, 2, 1)
plt.plot(fpr_list, tpr_list, color='darkorange', lw=2,
         label=f'ROC-кривая (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Случайный классификатор')
plt.scatter(opt_fpr, opt_tpr, color='red', s=100, zorder=5,
            label=f'Оптимальная точка\nFPR={opt_fpr:.4f}, TPR={opt_tpr:.4f}')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC-кривая байесовского классификатора спама')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# 2. Индекс Юдена в зависимости от порога
plt.subplot(2, 2, 2)
plt.plot(thresholds, youden_j, color='green', lw=2)
plt.axvline(x=opt_th, color='red', linestyle='--', alpha=0.7,
            label=f'Оптимальный порог = {opt_th:.2f}')
plt.xlabel('Порог ln(L_spam / L_legit)')
plt.ylabel('Индекс Юдена J = TPR - FPR')
plt.title('Поиск оптимального порога (максимум J)')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Распределение score для двух классов
plt.subplot(2, 2, 3)
plt.hist(scores[labels == 0], bins=30, alpha=0.7, color='blue',
         label='Легитимные письма', density=True)
plt.hist(scores[labels == 1], bins=30, alpha=0.7, color='red',
         label='Спам', density=True)
plt.axvline(x=opt_th, color='black', linestyle='--', lw=2,
            label=f'Оптимальный порог = {opt_th:.2f}')
plt.xlabel('ln(L_spam / L_legit)')
plt.ylabel('Плотность вероятности')
plt.title('Распределение отношения правдоподобия')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Зависимость TPR и FPR от порога
plt.subplot(2, 2, 4)
plt.plot(thresholds, tpr_list, color='green', lw=2, label='TPR')
plt.plot(thresholds, fpr_list, color='red', lw=2, label='FPR')
plt.axvline(x=opt_th, color='black', linestyle='--', lw=2,
            label=f'Оптимальный порог')
plt.xlabel('Порог ln(L_spam / L_legit)')
plt.ylabel('Значение')
plt.title('Зависимость TPR и FPR от порога')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("ЗАДАНИЕ 5 ВЫПОЛНЕНО УСПЕШНО!")
print("=" * 70)

print("\nКлючевые выводы по задаче 5:")
print("1. Дисбаланс классов (95% легитимных, 5% спама) существенно влияет на классификацию")
print("2. ROC-анализ позволяет выбрать оптимальный порог для конкретной задачи")
print(f"3. Оптимальный порог {opt_th:.4f} минимизирует ложные срабатывания")
print(f"4. AUC = {roc_auc:.4f} показывает отличную разделимость классов")
print("5. Для реальной системы нужно выбирать порог в зависимости от:")
print("   - Стоимости ложных срабатываний (потеря важных писем)")
print("   - Стоимости пропуска спама (неудобство пользователя)")
print("   - Процента спама в реальном трафике")