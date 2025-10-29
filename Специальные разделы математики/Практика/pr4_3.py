'''
Задание 6. Реализация операций в поле GF(2^3)
Условие:
• Используя базовое поле (F_2 = {0,1}) и неприводимый многочлен (x^3 + x + 1),
реализуйте класс для элементов поля GF(2^3).
• Определите операции сложения и умножения.
• Найдите произведение элементов (1 + 𝛼 + 𝛼 ^2) * (𝛼 + 𝛼 ^2).
'''
MOD = 2

class GF2_3:
    def __init__(self, a, b, c):
        self.a = a % MOD
        self.b = b % MOD
        self.c = c % MOD

    def __add__(self, other):
        return GF2_3(self.a ^ other.a, self.b ^ other.b, self.c ^ other.c)

    def __mul__(self, other):
        # Используем многочлен x^3 + x + 1, поэтому α^3 = α + 1
        # Умножение как многочленов с приведением по модулю 2
        coef = [0] * 5
        for i in range(3):
            for j in range(3):
                coef[i + j] ^= (self[i] * other[j])

        # Приведение по модулю α^3 + α + 1
        for i in range(4, 2, -1):
            if coef[i]:
                coef[i - 2] ^= coef[i]
                coef[i - 3] ^= coef[i]
                coef[i] = 0

        return GF2_3(coef[0], coef[1], coef[2])

    def __getitem__(self, idx):
        return [self.a, self.b, self.c][idx]

    def __str__(self):
        return f"{self.a} + {self.b}*α + {self.c}*α²"

if __name__ == "__main__":
    print("\nЗадание 6:")
    x = GF2_3(1, 1, 1)
    y = GF2_3(0, 1, 1)
    print(f"({x}) * ({y}) = {x * y}")

