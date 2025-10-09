'''
Задание 10. Исследование подгруппы в группе перестановок (S_4)
Условие:
Рассмотрите группу перестановок (S_4) — все перестановки множества {1,2,3,4}.
• Найдите подгруппу, состоящую из всех чётных перестановок (группа альтернаций
(A_4)).
• Определите порядок подгруппы.
'''

import itertools

def get_permutation_sign(perm):
    """
    Определяет знак перестановки (чётность)
    1 - чёт, -1 - нечёт
    """
    # Преобразуем перестановку в список (индексы с 0)
    n = len(perm)
    arr = list(perm)
    visited = [False] * n
    sign = 1

    for i in range(n):
        if not visited[i]:
            cycle_length = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = arr[j] - 1
                cycle_length += 1
            # Знак цикла длины k: (-1)^(k-1)
            if cycle_length > 1:
                sign *= (-1) ** (cycle_length - 1)
    return sign

def find_A_4_subgroup():
    # Генерируем все перестановки S4
    elements = [1, 2, 3, 4]
    S_4 = list(itertools.permutations(elements))

    print(f"Порядок группы S_4: {len(S_4)}")
    print("Все перестановки S_4:")
    for i, p in enumerate(S_4, 1):
        print(f"{i}: {p}")
    A_4 = []
    for perm in S_4:
        if get_permutation_sign(perm) == 1:  # Чётная перестановка
            A_4.append(perm)

    print(f"\nПорядок подгруппы A_4: {len(A_4)}")
    print("Чётные перестановки (A_4):")
    for i, p in enumerate(A_4, 1):
        print(f"{i}: {p}")

    # Проверяем, что A_4 действительно подгруппа (замкнутость)
    print("\nПроверка замкнутости A4 (композиция любых двух чётных перестановок даёт чётную):")
    for p1 in A_4:
        for p2 in A_4:
            # Композиция перестановок
            p3 = tuple(p1[p2[i] - 1] for i in range(len(p2)))
            if p3 not in A_4:
                print("A_4 не замкнута!")
                break
    print("A_4 замкнута относительно композиции - это подгруппа!")
    return A_4

if __name__ == "__main__":
    A_4 = find_A_4_subgroup()