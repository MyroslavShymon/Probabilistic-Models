import numpy as np
import matplotlib.pyplot as plt

# Задані дані
data_A = [
    [45, 51, 58, 76, 48],
    [59, 42, 62, 39, 51],
    [60, 66, 71, 73, 61],
    [46, 48, 50, 37, 34],
    [55, 53, 42, 26, 69],
    [41, 51, 36, 53, 68],
    [56, 46, 50, 38, 47],
    [49, 48, 52, 61, 48],
    [39, 58, 39, 36, 57],
    [58, 50, 42, 41, 66],
    [62, 64, 46, 41, 68],
    [65, 45, 46, 46, 49],
    [54, 52, 40, 42, 63],
    [41, 44, 55, 43, 46],
    [68, 59, 60, 60, 33]
]

data_B = [
    [49, 44, 51, 70, 54],
    [50, 35, 48, 23, 52],
    [45, 29, 37, 66, 44],
    [52, 66, 31, 59, 44],
    [47, 45, 53, 56, 59],
    [61, 64, 47, 54, 63],
    [53, 62, 56, 52, 51],
    [64, 36, 43, 52, 49],
    [47, 40, 35, 61, 38],
    [40, 55, 49, 62, 64],
    [49, 44, 51, 70, 54],
    [50, 35, 48, 23, 52],
    [45, 29, 37, 66, 44],
    [52, 66, 31, 59, 44],
    [47, 45, 53, 56, 59]
]

data_C = [
    [36, 64, 50, 67, 37],
    [48, 51, 54, 55, 28],
    [54, 47, 45, 57, 51],
    [46, 57, 50, 45, 54],
    [58, 35, 45, 65, 53],
    [55, 60, 42, 43, 65],
    [30, 47, 47, 41, 52],
    [49, 44, 57, 61, 54],
    [50, 47, 57, 52, 40],
    [69, 47, 50, 58, 58],
    [44, 42, 60, 44, 58],
    [44, 48, 52, 48, 56],
    [56, 63, 58, 52, 60],
    [36, 37, 42, 39, 38],
    [57, 55, 66, 61, 40]
]

data_D = [
    [42, 55, 51, 53, 58],
    [41, 30, 48, 54, 46],
    [50, 49, 62, 34, 35],
    [62, 41, 40, 38, 34],
    [63, 24, 41, 41, 46],
    [48, 57, 50, 53, 54],
    [31, 48, 55, 53, 60],
    [58, 63, 47, 42, 65],
    [53, 51, 43, 46, 57],
    [44, 53, 45, 54, 46],
    [35, 54, 42, 34, 49],
    [35, 36, 49, 37, 38],
    [42, 48, 34, 54, 51],
    [70, 39, 44, 41, 41],
    [50, 62, 43, 47, 49]
]

data_E = [
    [36, 64, 50, 67, 37],
    [48, 51, 54, 55, 28],
    [54, 47, 45, 57, 51],
    [46, 57, 50, 45, 54],
    [58, 35, 45, 65, 53],
    [55, 60, 42, 43, 65],
    [30, 47, 47, 41, 52],
    [49, 44, 57, 61, 54],
    [50, 47, 57, 52, 40],
    [69, 47, 50, 58, 58],
    [59, 72, 47, 39, 39],
    [54, 57, 39, 57, 49],
    [57, 59, 39, 45, 33],
    [70, 64, 49, 48, 62],
    [52, 55, 55, 60, 46]
]

def analyze_data(data):
    # Ранжування даних
    data = np.array(data).flatten()
    sorted_data = sorted(data)

    # Варіаційний ряд
    variation_series = sorted(list(set(sorted_data)))
    print('\nВаріаційний ряд:', variation_series)

    # Дискретний ряд
    values, frequencies_discrete = np.unique(sorted_data, return_counts=True)
    discrete_series = list(zip(values, frequencies_discrete))

    # Відносні частоти для дискретного ряду
    relative_frequencies_discrete = frequencies_discrete / len(sorted_data)

    # Інтервальний ряд
    hist, bin_edges = np.histogram(sorted_data, bins='auto')
    interval_series = list(zip(bin_edges[:-1], bin_edges[1:], hist))

    # Відносні частоти для інтервального ряду
    relative_frequencies_interval = hist / len(sorted_data)

    # Знаходження R*, Me*, Mo* для дискретного та інтервального рядів
    R_discrete = (sorted_data[-1] - sorted_data[0]) / (np.log10(len(sorted_data)) - np.log10(1))
    Me_discrete = np.median(sorted_data)
    max_frequency = max(frequencies_discrete)
    Mo_discrete = [value for value, frequency in discrete_series if frequency == max_frequency]
    print('\nВибірковий розмах дискретного ряду:', R_discrete)
    print('Медіана (Me*) дискретного ряду:', Me_discrete)
    if len(Mo_discrete) == 1:
        print('Мода (Mo*) дискретного ряду:', Mo_discrete[0])
    else:
        print('Моди дискретного ряду:', ', '.join(map(str, Mo_discrete)))

    interval_width = interval_series[0][1] - interval_series[0][0]
    R_interval = (interval_series[-1][1] - interval_series[0][0]) / (np.log10(len(sorted_data)) - np.log10(1))
    Me_interval = interval_series[len(interval_series) // 2][0] + interval_width / 2
    Mo_interval = max(interval_series, key=lambda x: x[2])
    N = sum(interval[2] for interval in interval_series)
    Mo_interval = Mo_interval[0] + interval_width * (Mo_interval[2] - Mo_interval[1]) / (2 * Mo_interval[2] - Mo_interval[1] - Mo_interval[0])
    print('\nІнтервальний ряд:')
    for interval in interval_series:
        print('Інтервал [{:.2f}, {:.2f}], Абсолютна частота: {}'.format(interval[0], interval[1], interval[2]))
    print('Відносна частота інтервального ряду:', relative_frequencies_interval)

    print('\nВибірковий розмах (R*) інтервального ряду:', R_interval)
    print('Медіана (Me*) інтервального ряду:', Me_interval)
    print('Мода (Mo*) інтервального ряду:', Mo_interval)

    # Побудова гістограми та полігон частот
    plt.plot(values, relative_frequencies_discrete, marker='o')
    plt.title('Полігон відносних частот')
    plt.xlabel('Значення')
    plt.ylabel('Відносна частота')
    plt.grid(True)
    plt.show()

    # Побудова гістограми для інтервального ряду
    x = np.array([interval[0] for interval in interval_series])
    y = np.array(relative_frequencies_interval)
    width = bin_edges[1] - bin_edges[0]
    plt.bar(x, y, width=width, align='edge', edgecolor='black')
    plt.xlabel('Інтервали')
    plt.ylabel('Відносна частота')
    plt.show()

    # Запишемо функції для знаходження вибіркового середнього, S^2 та коефіцієнту варіації:
    x_bar = np.mean(sorted_data)
    s_squared = np.var(sorted_data)
    V = s_squared / x_bar
    print('\nВибіркове середнє (x_bar):', x_bar)
    print('Вибіркова дисперсія (S^2):', s_squared)
    print('Коефіцієнт варіації (V):', V)

# Вибір даних для аналізу
choice = input("Які дані ви бажаєте проаналізувати? Введіть 'A', 'B', 'C', 'D', 'E': ").upper()

if choice == 'A':
    analyze_data(data_A)
elif choice == 'B':
    analyze_data(data_B)
elif choice == 'C':
    analyze_data(data_C)
elif choice == 'D':
    analyze_data(data_D)
elif choice == 'E':
    analyze_data(data_E)
else:
    print("Неправильний вибір.")