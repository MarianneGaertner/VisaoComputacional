import cv2
import numpy as np
from matplotlib import pyplot as plt

I = cv2.imread('./Imagens/monalisa2.png', cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(I, cmap='gray')

# transformação de translação
dx = 40
dy = 20

A = np.array([[1, 0, dx], [0, 1, dy]], np.float32)

n_linhas, n_colunas = I.shape
n_linhas_final = n_linhas + dy
n_colunas_final = n_colunas + dx

I2 = cv2.warpAffine(I, A, (n_colunas_final, n_linhas_final))

plt.figure()
plt.imshow(I2, cmap='gray')

plt.show()
