import numpy as np
import scipy.stats as stats

# Вхідні дані
data = np.array([
    [453, 19, 171, 511, 2801],
    [427, 20, 180, 471, 2715],
    [377, 15, 96, 499, 2793],
    [396, 17, 140, 523, 2527],
    [403, 23, 89, 483, 2720],
    [399, 21, 92, 498, 2735],
    [405, 15, 214, 498, 2572],
    [418, 18, 173, 542, 2817],
    [389, 17, 216, 463, 2639],
    [413, 22, 87, 501, 2736],
    [402, 15, 125, 539, 2543],
    [412, 20, 93, 471, 2682],
    [396, 21, 125, 492, 2828],
    [423, 17, 210, 523, 2593],
    [439, 19, 217, 463, 2702]
])

# Розділення даних на Y (залежна змінна) та X (незалежні змінні)
Y = data[:, 0]
X = data[:, 1:]

# Крок 1: Знайти точкові незміщені статистичні оцінки b0*, b1*, b2*, b3*, b4*
X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ Y

beta0 = beta[0]
beta1 = beta[1]
beta2 = beta[2]
beta3 = beta[3]
beta4 = beta[4]

print("beta0*:", beta0)
print("beta1*:", beta1)
print("beta2*:", beta2)
print("beta3*:", beta3)
print("beta4*:", beta4)

# Крок 2: Побудувати довірчий інтервал для функції регресії з надійністю у = 0.99
n = len(Y)
p = X.shape[1]

gamma = 0.99
t_value = stats.t.ppf(1 - (1 - gamma) / 2, n - p - 1) # двобічний довірчий інтервал

Y_pred = X_with_intercept @ beta
residuals = Y - Y_pred
residual_var = np.sum(residuals ** 2) / (n - p- 1)

# обчислення вектора залишкових помилок
e = Y - Y_pred

# Обчислення стандартного відхилення
SE = np.sqrt(np.sum(e ** 2) / (n - p - 1))

# обчислення довірчого інтервалу
CI_lower = np.mean(Y_pred) - t_value * SE
CI_upper = np.mean(Y_pred) + t_value * SE

print("Довірчий інтервал для функції регресії:")
print("CI_y=", (CI_lower, CI_upper))


# Крок 3: Обчислити коефіцієнт множинної кореляції в
Y_mean = np.mean(Y)
SS_total = np.sum((Y - Y_mean) ** 2)
SS_reg = np.sum((Y_pred - Y_mean) ** 2)
R = np.sqrt(SS_reg / SS_total)

print("Коефіцієнт множинної кореляції R:", R)