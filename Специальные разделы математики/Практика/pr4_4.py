'''
Задание 7. Исследование свойств поля GF(11)
Условие:
• Напишите программу, которая реализует операции сложения, вычитания,
умножения и деления в поле простого порядка 11 (F_{11}).
• Проверьте, что для каждого ненулевого элемента существует мультипликативный
обратный.
• Для элемента 7 найдите обратный и проверьте результат.
'''
MOD = 11
class GF11:
    def __init__(self, val):
        self.val = val % MOD

    def __add__(self, other):
        return GF11(self.val + other.val)

    def __sub__(self, other):
        return GF11(self.val - other.val)

    def __mul__(self, other):
        return GF11(self.val * other.val)

    def inverse(self):
        for i in range(1, MOD):
            if (self.val * i) % MOD == 1:
                return GF11(i)
        raise ValueError("Обратного элемента не существует")

    def __str__(self):
        return str(self.val)

if __name__ == "__main__":
    print("\nЗадание 7:")
    elem_7 = GF11(7)
    inv_7 = elem_7.inverse()
    print(f"Обратный к {elem_7}: {inv_7}")
    print(f"Проверка: {elem_7} * {inv_7} = {elem_7 * inv_7}")
