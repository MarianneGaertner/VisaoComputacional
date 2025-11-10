import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Produto de matrizes
produto = np.dot(A, B)
print("Produto de Matrizes:\n", produto)

# Resolução de Sistemas Lineares
b = np.array([9, 8])
x = np.linalg.solve(A, b)
print("Solução do sistema:", x)