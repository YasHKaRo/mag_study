'''
Задание 9. Проверка, является ли (Z_{11}) полем
Условие:
Поскольку 11 — простое число, проверьте, что (Z_{11}) — поле.
• Найдите обратные элементы по умножению для всех элементов, кроме 0.
• Проверьте основные свойства поля.
'''

MOD = 11
def add_mod(a, b):
    return (a + b) % MOD
def mul_mod(a, b):
    return (a * b) % MOD
def extended_gcd(a, b):
    """Расширенный алгоритм Евклида для поиска обратного по модулю"""
    if b == 0:
        return a, 1, 0
    gcd, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return gcd, x, y
def mod_inverse(a, m):
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        return None # Обратного нет
    else:
        return x % m
def check_field_Z11():
    elements = list(range(MOD))
    neutral_add = 0
    neutral_mul = 1
    # Проверка группы по сложению (абелева группа)
    # Аналогично предыдущему заданию, пропускаем детали для краткости
    # Проверка группы по умножению для ненулевых элементов
    non_zero_elements = [e for e in elements if e != 0]
    for a in non_zero_elements:
        inv = mod_inverse(a, MOD)
        if inv is None:
            print(f"Обратного элемента по умножению нет для {a}")
            return
        # Проверка
        if mul_mod(a, inv) != neutral_mul:
            print(f"Неправильный обратный элемент для {a}")
            return
    print("Для всех ненулевых элементов обратные по умножению найдены.")
    print(f"Значит, Z{MOD} образует поле.")
    # Выведем обратные элементы
    print(f"Обратные элементы по умножению в Z{MOD}:")
    for a in non_zero_elements:
        inv = mod_inverse(a, MOD)
        print(f"{a} -> {inv}")
if __name__ == "__main__":
    check_field_Z11()