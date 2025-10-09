def lfsr(taps, state, length):
    """
    taps: список позиций обратной связи ([5,2])
    state: начальное состояние (список битов)
    length: длина генерируемой последовательности
    """
    output = []
    for _ in range(length):
        print(f"Шаг {_ + 1}: {state}")
        output.append(state[-1])  # выходной бит — последний в регистре
        # вычисляем новый бит как XOR битов на позициях taps
        new_bit = 0
        for tap_index in taps:
            new_bit ^= state[-tap_index]
        state = [new_bit] + state[:-1]
        # print(state)
    return output
if __name__ == "__main__":
    taps = [4, 3]
    init_state = [1, 1, 0, 0]
    seq_length = 15
    sequence = lfsr(taps, init_state, seq_length)
    print("Сгенерированная последовательность:")
    print(sequence)

