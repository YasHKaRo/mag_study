'''
Задание 5. Нахождение обратного элемента в поле GF(5^2)
Условие:
Для поля GF(5^2) с неприводимым многочленом (x^2 + x + 2) (проверьте неприводимость
самостоятельно):
• Запишите соотношение для (𝛾^2), где (𝛾) — корень многочлена.
• Найдите обратный элемент для (2 + 3𝛾).
• Проверьте результат, умножив исходный элемент на найденный обратный.
'''


MOD = 5

class GF5_2:
    def __init__(self, a, b):
        self.a = a % MOD
        self.b = b % MOD
    def __mul__(self, other):
        # Используем многочлен x^2 + x + 2, поэтому y^2 = -y - 2 = 4y + 3
        a = (self.a * other.a + 3 * self.b * other.b) % MOD
        b = (self.a * other.b + self.b * other.a + 4 * self.b * other.b) % MOD
        return GF5_2(a, b)
    def inverse(self):
        for x in range(MOD):
            for y in range(MOD):
                candidate = GF5_2(x, y)
                if (self * candidate) == GF5_2(1, 0):
                    return candidate
        raise ValueError("Обратного элемента не существует")
    def __str__(self):
        return f"{self.a} + {self.b}*y"
if __name__ == "__main__":
    print("\nЗадание 5:")
    elem = GF5_2(2, 3)
    inv_elem = elem.inverse()
    print(f"Обратный к {elem}: {inv_elem}")
    print(f"Проверка: {elem} * {inv_elem} = {elem * inv_elem}")

