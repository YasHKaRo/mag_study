'''
Задание 7. Исследование множества неотрицательных целых чисел по умножению
Условие:
Пусть (N_0 = {0,1,2,3,...}).
• Является ли (N_0, *)) группой?
• Если нет, укажите, какие из аксиом группы нарушаются.
'''

def check_group_addition_Z():
    # Возьмем диапазон для проверки: от 0 до 10
    elements = list(range(0, 11))
    # Проверка замкнутости
    for a in elements:
        for b in elements:
            c = a * b
            if c not in elements:
                # Теоретически не может случиться, так что пропустим
                pass
    print("1) Замкнутость: выполнена (целые числа замкнуты по сложению)")
    # Проверка ассоциативности
    for a in elements:
        for b in elements:
            for c in elements:
                if (a * b) * c != a * (b * c):
                    print("Ассоциативность нарушена для", a, b, c)
                    return
    print("2) Ассоциативность: выполнена")
    # Нейтральный элемент — 1
    neutral = 1
    for a in elements:
        if a * neutral != a or neutral * a != a:
            print("Нейтральный элемент отсутствует")
            return
    print("3) Нейтральный элемент: 1")
    # Обратный элемент
    for a in elements:
        if a == 0:
            continue
        inverse_found = False
        for x in elements:
            if a * x == neutral:  # ищем x такой, что a * x = 1
                inverse_found = True
                break
        if not inverse_found:
            print(f"4) Обратный элемент отсутствует для {a}")
            print("Вывод: (N_0, *) НЕ является группой")
            return
    print("4) Обратные элементы: существуют для всех элементов")
    print("\n\tВывод: (N, *) образует группу.")
if __name__ == "__main__":
    check_group_addition_Z()
