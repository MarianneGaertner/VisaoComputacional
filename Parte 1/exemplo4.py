import cv2
import numpy as np
from matplotlib import pyplot as plt

I = cv2.imread('./Imagens/quadrados_cinzas.png', cv2.IMREAD_GRAYSCALE)

print('Dados tipo:', I.dtype)

# Tamanho da Imagem
n_linhas, n_colunas = I.shape
print('Número de linhas: ', n_linhas)
print('Número de colunas: ', n_colunas)

print(I[100,100])
print(I[100,300])
print(I[300,100])
print(I[300,300])

plt.figure()
plt.imshow(I, cmap='gray')
plt.show()
