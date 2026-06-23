import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
# Periodic function: half sine wave
def f(x):
    x_mod = (x + np.pi) % (2*np.pi) - np.pi
    return np.sin(x_mod) if x_mod >= 0 else 0.0
# Vectorized for plotting
f_vec = np.vectorize(f)
# Fourier coefficients
def a_n(n):
    integrand = lambda x: f(x) * np.cos(n*x)
    return (1/np.pi) * quad(integrand, -np.pi, np.pi)[0]
def b_n(n):
    integrand = lambda x: f(x) * np.sin(n*x)
    return (1/np.pi) * quad(integrand, -np.pi, np.pi)[0]
def a0():
    integrand = lambda x: f(x)
    return (1/np.pi) * quad(integrand, -np.pi, np.pi)[0]
# Fourier partial sum
def fourier_series(x, N):
    s = a0()/2
    for n in range(1, N+1):
        s += a_n(n)*np.cos(n*x) + b_n(n)*np.sin(n*x)
    return s
# Infinite sum with tolerance
def fourier_series_tol(x, tol=0.0001, max_terms=1000):
    s = a0()/2
    term = 1
    n = 1
    while abs(term) > tol and n < max_terms:
        term = a_n(n)*np.cos(n*x) + b_n(n)*np.sin(n*x)
        s += term
        n += 1
    return s
# Plotting
x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = f_vec(x)
plt.figure(figsize=(8,6))
plt.plot(x, y, label="Original Function")
# Plot partial sums
In [4]:
for N in [1, 3, 5, 10, 20]:
yN = [fourier_series(val, N) for val in x]
plt.plot(x, yN, label=f"N={N}")
plt.title("Fourier Reconstruction of Half Sine Wave")
plt.legend()
plt.grid(True)
plt.show()