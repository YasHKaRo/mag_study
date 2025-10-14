def find_period(taps, init_state):
    state = init_state.copy()
    initial = tuple(state)
    count = 0
    seen_states = set()
    while True:
        count += 1
        new_bit = 0
        for t in taps:
            new_bit ^= state[-t]
        state = [new_bit] + state[:-1]
        state_tuple = tuple(state)
        if state_tuple == initial:
            return count
        if state_tuple in seen_states:
            # Если встретили другое повторение — период меньше
            return count
        seen_states.add(state_tuple)

list_of_numb = []
for i in range(8):
    bit_numb = bin(i)
    bit_numb = bit_numb[2:]
    list_numb = []
    for symb in bit_numb:
        list_numb.append(symb)
    while len(list_numb) < 4:
        list_numb = ['0'] + list_numb
    for i in range(len(list_numb)):
        list_numb[i] = int(list_numb[i])
    list_of_numb.append(list_numb)

if __name__ == "__main__":
    pull_numb = [1, 2, 3, 4]
    pull_taps = []
    count = 0
    for numb_1 in pull_numb[::-1]:
        count += 1
        for numb_2 in pull_numb[::-1]:
            if numb_1 != numb_2:
                if ([numb_1, numb_2] not in pull_taps) and (numb_1 > numb_2):
                    pull_taps.append([numb_1, numb_2])

    for tap in pull_taps:
        print(tap)
        for init_state in list_of_numb:
            period = find_period(tap, init_state)
            print(f"Период LFSR: {period} Последовательность: {init_state}")
