import numpy as np

# Geração de números aleatórios uniformes entre 0 e 1
arr_uniform = np.random.rand(3, 3)
print("Array com distribuição uniforme [0, 1):\n", arr_uniform)

# Geração de números aleatórios com distribuição normal (média 0, desvio padrão 1)
arr_normal = np.random.randn(3, 3)
print("Array com distribuição normal (média 0, desvio padrão 1):\n", arr_normal)

# Geração de números inteiros aleatórios entre 1 e 10
arr_inteiros = np.random.randint(1, 10, (3, 3))
print("Array com inteiros aleatórios entre 1 e 10:\n", arr_inteiros)