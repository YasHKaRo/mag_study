import matplotlib.pyplot as plt
def lfsr(taps, state, length):
    n = len(state)
    output = []
    for _ in range(length):
        output.append(state[-1])
        new_bit = 0
        for t in taps:
            new_bit ^= state[-t]
        state = [new_bit] + state[:-1]
    return output
if __name__ == "__main__":
    taps = [6, 1]
    init_state = [1, 0, 0, 0, 0, 0]
    length = 63
    seq = lfsr(taps, init_state, length)
    print("Последовательность LFSR:")
    print(seq)
    print(f"Количество единиц: {seq.count(1)}")
    print(f"Количество нулей: {seq.count(0)}")
    counts = [seq.count(0), seq.count(1)]
    plt.bar(['0', '1'], counts)
    plt.show()