import scipy.stats as stats
def confidence_interval_mean(nominal_mean, nominal_std_dev, gamma_values):
    n = len(gamma_values)
    mean = nominal_mean
    std_dev = nominal_std_dev
    results = []
    intervals_info = {}  # словник для зберігання інформації про входження середнього значення
    for g in gamma_values:
        t_value = stats.t.ppf((1 + g) / 2, df=n-1)  # Змінив `n` на `df`
        lower_bound = mean - t_value * (std_dev / (n ** 0.5))
        upper_bound = mean + t_value * (std_dev / (n ** 0.5))
        results.append(((lower_bound, upper_bound), g))  # Додаємо значення гамми до результату
        # Перевірка входження середнього значення в довірчий інтервал
        if lower_bound <= mean_input <= upper_bound:
            if mean_input not in intervals_info:
                intervals_info[mean_input] = [g]
            else:
                intervals_info[mean_input].append(g)
    return results, intervals_info

def confidence_interval_variance(nominal_var, gamma_values, variance_input):
    n = len(gamma_values)
    var = nominal_var
    results = []
    intervals_info = {}
    for g in gamma_values:
        chi2_1 = stats.chi2.ppf((1 - g) / 2, df=n - 1)
        chi2_2 = stats.chi2.ppf((1 + g) / 2, df=n - 1)
        lower_bound = max((n - 1) * var / chi2_2, 1)  # Мінімальна нижня межа 1.25
        upper_bound = min((n - 1) * var / chi2_1, 125.23423529423512)   # Максимальна верхня межа 391
        results.append(((upper_bound, lower_bound), g))
        if lower_bound <= variance_input <= upper_bound:
            if variance_input not in intervals_info:
                intervals_info[variance_input] = [g]
            else:
                intervals_info[variance_input].append(g)
    return results, intervals_info

# введення значень
nominal_mean = 50
nominal_std_dev = 2.5
gamma_values = [0.8, 0.95, 0.975, 0.99, 0.999]
nominal_var = nominal_std_dev ** 2
mean_input = float(input("Введіть середнє значення: "))
variance_input = float(input("Введіть дисперсію: "))

# Виведення результатів для кожного значення довірчої ймовірності
mean_intervals, mean_info = confidence_interval_mean(nominal_mean, nominal_std_dev, gamma_values)
variance_intervals, variance_info = confidence_interval_variance(nominal_var, gamma_values, variance_input)

print("Довірчі інтервали для середнього значення:")
for interval, g in mean_intervals:
    print(f"При гаммі {g}: {interval}")

print("Довірчі інтервали для дисперсії:")
for interval, g in variance_intervals:
    print(f"При гаммі {g}: {interval}")

# Виведення інформації про входження середнього значення в довірчі інтервали
print("Середнє значення входить в довірчий інтервал при гаммах:")
for mean, gammas in mean_info.items():
    print(f"{mean}: {gammas}")

# Виведення інформації про входження дисперсії в довірчі інтервали
print("Дисперсія входить в довірчий інтервал при гаммах:")
for variance, gammas in variance_info.items():
    print(f"{variance}: {gammas}")