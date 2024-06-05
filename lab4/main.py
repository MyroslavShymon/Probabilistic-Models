import numpy as np
from scipy.stats import norm, chi2, kstest
import matplotlib.pyplot as plt

# Дані
xi_intervals = [(25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33)]
mi = [10, 8, 7, 10, 22, 18, 14, 11]
alpha = 0.05
n = np.sum(mi)  # Загальна кількість спостережень

# Критерій Пірсона
print("\nКритерій Пірсона:")

# Розрахунок очікуваних частот за допомогою емпіричної функції розподілу
empirical_cdf_values = np.cumsum(mi) / n
mu_empirical = np.diff(np.concatenate(([0], empirical_cdf_values))) * n

# Виведення очікуваних частот (mu_empirical) та емпіричної функції розподілу (empirical_cdf_values)
print("Емпіричні частоти (mi):", mi)
print("Емпірична функція розподілу (CDF):", empirical_cdf_values)
print("Очікувані частоти (mu_empirical):", mu_empirical)

# Розрахунок статистики критерію Пірсона
chi_square_statistic = np.sum((np.array(mi) - mu_empirical) ** 2 / mu_empirical)

# Розрахунок критичного значення
df = len(mi) - 1
critical_value = chi2.ppf(1 - alpha, df)

# Виведення результатів
print("Статистика критерію Пірсона:", chi_square_statistic)
print("Критичне значення:", critical_value)
if chi_square_statistic > critical_value:
    print("\nГіпотеза H0 відхиляється")
else:
    print("\nГіпотеза H0 НЕ відхиляється")

# Критерій Колмогорова
print("\nКритерій Колмогорова:")

# Знаходимо всі значення xi, які відповідають середнім точкам кожного інтервалу
xi_values = [(interval[0] + interval[1]) / 2 for interval in xi_intervals]

# Виведення значень xi
print("Середні точки інтервалів (xi):", xi_values)

# Розрахунок статистики критерію Колмогорова
ks_statistic, p_value = kstest(xi_values, 'norm', args=(np.mean(xi_values), np.std(xi_values)))

# Розрахунок критичного значення
critical_value_ks = np.sqrt(-0.5 * np.log(alpha / 2)) / np.sqrt(n)

# Виведення результатів
print("Статистика критерію Колмогорова:", ks_statistic)
print("Критичне значення:", critical_value_ks)
print("p-значення:", p_value)
if ks_statistic > critical_value_ks:
    print("\nГіпотеза H0 відхиляється")
else:
    print("\nГіпотеза H0 НЕ відхиляється")

# Графічна перевірка
print("\nГрафічна перевірка:")

# Емпірична функція розподілу
empirical_cdf = np.cumsum(mi) / n

# Теоретична функція розподілу нормального розподілу
theoretical_cdf = norm.cdf(xi_values, loc=np.mean(xi_values), scale=np.std(xi_values))

# Виведення емпіричної та теоретичної функцій розподілу
print("Емпірична функція розподілу:", empirical_cdf)
print("Теоретична функція розподілу:", theoretical_cdf)

# Графік
plt.plot(xi_values, empirical_cdf, label='Емпірична ФР')
plt.plot(xi_values, theoretical_cdf, label='Теоретична ФР')
plt.xlabel('Значення xi')
plt.ylabel('Ймовірність')
plt.title('Графічна перевірка нормального розподілу')
plt.legend()
plt.grid(True)
plt.show()