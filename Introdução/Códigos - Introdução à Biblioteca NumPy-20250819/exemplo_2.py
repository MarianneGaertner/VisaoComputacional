import numpy as np

# Criação a partir de uma lista
lista = [1, 2, 3, 4, 5]
arr = np.array(lista)
print(arr)

# Criação usando funções do NumPy
arr_zeros = np.zeros((3, 3))  # Array 3x3 de zeros
arr_uns = np.ones((2, 2))     # Array 2x2 de uns
arr_range = np.arange(10)     # Array de 0 a 9
print(arr_zeros)
print(arr_uns)
print(arr_range)