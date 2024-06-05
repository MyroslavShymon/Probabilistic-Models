import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Набір даних
Y = np.array([29, 38, 49, 54, 62, 70, 79, 98])
X = np.array([15.99, 19.75, 23.10, 26.44, 29.79, 33.13, 36.89, 44.54])

# Крок 1: Знайти точкові незміщені статистичні оцінки β0*, β1*
beta1 = np.cov(X, Y, bias=True)[0, 1] / np.var(X)
beta0 = np.mean(Y) - beta1 * np.mean(X)
print("Крок 1: Точкові незміщені статистичні оцінки")
print("β0*:", beta0)
print("β1*:", beta1)

# Побудова графіку розсіювання
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', label='Спостереження')

# Побудова лінії регресії
plt.plot(X, beta0 + beta1 * X, color='red', label='Лінія регресії')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Графік розсіювання та лінія регресії')
plt.legend()
plt.grid(True)
plt.show()

# Крок 2: Побудувати довірчі інтервали для параметрів β0*, β1* з надійністю γ = 0.99
n = len(X)
gamma = 0.99
t_value = stats.t.ppf(1 - (1 - gamma) / 2, n - 2) # двобічний довірчий інтервал
print("\nКрок 2: Розрахунок довірчих інтервалів")
print("Кількість спостережень (n):", n)
print("Критичне значення t-розподілу:", t_value)

residuals = Y - (beta0 + beta1 * X)
print("Залишки (residuals):", residuals)

# Довірчий інтервал для β0
SE_beta0 = np.sqrt(np.sum(residuals ** 2) / (n - 2) * (1 / n + np.mean(X) ** 2 / np.sum((X - np.mean(X)) ** 2)))
print("Стандартна помилка для β0 (SE_beta0):", SE_beta0)
CI_beta0 = (beta0 - t_value * SE_beta0, beta0 + t_value * SE_beta0)
print("Довірчий інтервал для β0:", CI_beta0)

# Довірчий інтервал для β1
SE_beta1 = np.sqrt(np.sum(residuals ** 2) / ((n - 2) * np.sum((X - np.mean(X)) ** 2)))
print("Стандартна помилка для β1 (SE_beta1):", SE_beta1)
CI_beta1 = (beta1 - t_value * SE_beta1, beta1 + t_value * SE_beta1)
print("Довірчий інтервал для β1:", CI_beta1)

# Крок 3: Перевірити значущість параметра β1 при рівні значущості α = 0.01
print("\nКрок 3: Перевірка значущості параметра β1")
alpha = 0.01
t_statistic_beta1 = beta1 / SE_beta1
critical_t_value = stats.t.ppf(1 - alpha / 2, n - 2)
print("t-статистика для β1:", t_statistic_beta1)
print("Критичне значення t-статистики:", critical_t_value)

if np.abs(t_statistic_beta1) > critical_t_value:
    print("β1 є значущим")
else:
    print("β1 не є значущим")

# Крок 4: Побудувати довірчий інтервал для функції регресії yi = β0 + β1xi з надійністю γ = 0.99
print("\nКрок 4: Довірчий інтервал для функції регресії")
Y_pred = beta0 + beta1 * X
print("Прогнозовані значення (Y_pred):", Y_pred)

residuals = Y - Y_pred
print("Оновлені залишки (residuals):", residuals)

SE_Y_pred = np.sqrt((np.sum(residuals ** 2) / (n - 2)) * (1 + 1 / n + (X - np.mean(X)) ** 2 / np.sum((X - np.mean(X)) ** 2)))
print("Стандартна помилка прогнозованих значень (SE_Y_pred):", SE_Y_pred)

CI_Y_pred = (Y_pred - t_value * SE_Y_pred, Y_pred + t_value * SE_Y_pred)
print("Довірчий інтервал для функції регресії:")
for i in range(len(X)):
    print(f"x = {X[i]}, CI_y = {CI_Y_pred[0][i]} - {CI_Y_pred[1][i]}")

# Крок 5: Обчислити вибірковий коефіцієнт кореляції
print("\nКрок 5: Вибірковий коефіцієнт кореляції")
r = np.corrcoef(X, Y)[0, 1]
print("Вибірковий коефіцієнт кореляції r:", r)


# Побудова графіку розсіювання
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', label='Спостереження')

# Побудова лінії регресії
plt.plot(X, beta0 + beta1 * X, color='red', label='Лінія регресії')

# Додавання кореляційної функції
plt.text(20, 80, f'r = {r:.2f}', fontsize=12)

# Додавання довірчих інтервалів
for i in range(len(X)):
    plt.vlines(X[i], CI_Y_pred[0][i], CI_Y_pred[1][i], color='green', linestyle='dashed', linewidth=1)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Графік розсіювання, лінія регресії, кореляційна функція та довірчі інтервали')
plt.legend()
plt.grid(True)
plt.show()