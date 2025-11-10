import numpy as np
import cv2
from matplotlib import pyplot as plt

# 0->preto, 255->branco
M = 200  # linhas
N = 400  # colunas

I = np.zeros((M, N), np.float32)

I[50:150,20:380] = 1


# apresentar a imagem na tela
# cv2.imshow('Imagem binária', I)
# cv2.waitKey(0) # o programa fica parado aqui até alguma tecla ser pressionada

plt.figure()
plt.imshow(I, cmap='gray')
plt.show()
