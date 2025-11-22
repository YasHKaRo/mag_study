def weak_hash(text):
    return sum(ord(c) for c in text) % 256
def find_preimage(target_hash, max_length=4, charset='abcdefghijklmnopqrstuvwxyz'):
    """
    Поиск прообраза для заданного хеша перебором
    Args:
        target_hash: целевое значение хеша (0-255)
        max_length: максимальная длина строки для перебора
        charset: набор символов для перебора
    """
    from itertools import product
    print(f"Поиск строки с хешем = {target_hash}")
    total_tested = 0
    found = []
    # Перебор строк разной длины
    for length in range(1, max_length + 1):
        # Генерируем все комбинации длины length
        for chars in product(charset, repeat=length):
            test_string = ''.join(chars)
            hash_value = weak_hash(test_string)
            total_tested += 1
            if hash_value == target_hash:
                found.append(test_string)
                print(f"НАЙДЕНО: '{test_string}' -> хеш = {hash_value}")
    print(f"Всего проверено строк: {total_tested}")
    if found:
        print(f"Найдено прообразов: {len(found)}")
        print(f"Примеры: {found[:5]}")  # Показываем первые 5 найденных
    else:
        print("Прообраз не найден в заданном диапазоне")
    return found

# Демонстрация работы
if __name__ == "__main__":
    target = 195  # Целевой хеш
    preimages = find_preimage(target, max_length=3)