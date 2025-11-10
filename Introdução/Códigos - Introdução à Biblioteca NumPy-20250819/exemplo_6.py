import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Indexação
print("Primeiro elemento:", arr[0])
print("Último elemento:", arr[-1])

# Fatiamento
print("Dois primeiros elementos:", arr[:2])
print("Elementos do meio:", arr[1:4])
print("Todos os elementos exceto os dois últimos:", arr[:-2])