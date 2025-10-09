'''
Задание 8. Проверка, является ли множество вычетов (Z_{10}) кольцом
Условие:
Для множества (Z_{10} = {0,1,...,9}) с операциями сложения и умножения по модулю 10:
• Проверьте свойства кольца (замкнутость, ассоциативность, дистрибутивность).
• Определите наличие делителей нуля.
'''

MOD = 10
def add_mod(a, b):
    return (a + b) % MOD
def mul_mod(a, b):
    return (a * b) % MOD
def check_ring_Z10():
    elements = list(range(MOD))
# Проверяем абелеву группу по сложению
    neutral_add = 0
# Замкнутость по сложению
    for a in elements:
        for b in elements:
            if add_mod(a, b) not in elements:
                print("Не замкнуто по сложению")
                return
    print("Сложение замкнуто")
    # Ассоциативность по сложению
    for a in elements:
        for b in elements:
            for c in elements:
                left = add_mod(add_mod(a, b), c)
                right = add_mod(a, add_mod(b, c))
                if left != right:
                    print("Сложение не ассоциативно")
                    return
    print("Сложение ассоциативно")
# Нейтральный элемент по сложению
    for a in elements:
        if add_mod(a, neutral_add) != a or add_mod(neutral_add, a) != a:
            print("Нет нейтрального элемента по сложению")
            return
    print("Нейтральный элемент по сложению:", neutral_add)
# Обратный элемент по сложению
    for a in elements:
        found_inverse = False
        for x in elements:
            if add_mod(a, x) == neutral_add:
                found_inverse = True
                break
        if not found_inverse:
            print(f"Нет обратного элемента по сложению для {a}")
            return
    print("Обратные элементы по сложению: есть для всех")
    # Проверка умножения — ассоциативность и дистрибутивность (частично)
    # Ассоциативность по умножению
    for a in elements:
        for b in elements:
            for c in elements:
                if mul_mod(mul_mod(a, b), c) != mul_mod(a, mul_mod(b, c)):
                    print("Умножение не ассоциативно")
                    return
    print("Умножение ассоциативно")
# Дистрибутивность слева
    for a in elements:
        for b in elements:
            for c in elements:
                left = mul_mod(a, add_mod(b, c))
                right = add_mod(mul_mod(a, b), mul_mod(a, c))
                if left != right:
                    print("Дистрибутивность слева не выполняется")
                    return
    print("Дистрибутивность слева выполнена")
# Дистрибутивность справа
    for a in elements:
        for b in elements:
            for c in elements:
                left = mul_mod(add_mod(b, c), a)
                right = add_mod(mul_mod(b, a), mul_mod(c, a))
                if left != right:
                    print("Дистрибутивность справа не выполняется")
                    return
    print("Дистрибутивность справа выполнена")
# Проверка на делители нуля
    zero = 0
    zero_divisors = []
    for a in elements:
        for b in elements:
            if a != zero and b != zero and mul_mod(a, b) == zero:
                zero_divisors.append((a, b))
    if zero_divisors:
        print("Найдены делители нуля:", zero_divisors)
    else:
        print("Делителей нуля нет")
if __name__ == "__main__":
    check_ring_Z10()