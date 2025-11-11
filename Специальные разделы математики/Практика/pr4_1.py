'''
Задание 4: Построение поля GF(7^2)
Условие:
• Выберите неприводимый многочлен степени 2 над (F_7).
• Запишите все элементы поля GF(7^2) в полиномиальной форме (a + b𝛾), где (𝛾) —
корень выбранного многочлена.
• Определите правило умножения (выразите (𝛾^2) через 𝛾 и константу).
• Найдите произведение элементов (3 + 5𝛾) * (4 + 6𝛾).
'''
MOD_7 = 7

class GF7_2:
    def __init__(self, a, b):
        self.a = a % MOD_7
        self.b = b % MOD_7
    def __add__(self, other):
        return GF7_2(self.a + other.a, self.b + other.b)
    def __sub__(self, other):
        return GF7_2(self.a - other.a, self.b - other.b)
    def __mul__(self, other):
        # Используем неприводимый многочлен x^2 + 1, поэтому y^2 = -1 = 6
        # (a + by)*(a + by) = a*a +a*by +a*by +by*by = (a*a + b*6) + (2*aby)
        a = (self.a * other.a + 6 * self.b * other.b) % MOD_7
        b = (self.a * other.b + self.b * other.a) % MOD_7
        return GF7_2(a, b)
    def __eq__(self, other):
        return self.a == other.a and self.b == other.b
    def __str__(self):
        return f"{self.a} + {self.b}*y"

if __name__ == "__main__":
    print("Задание 4:")
    x = GF7_2(3, 5)
    y = GF7_2(4, 6)
    print(f"({x}) * ({y}) = {x * y}")

