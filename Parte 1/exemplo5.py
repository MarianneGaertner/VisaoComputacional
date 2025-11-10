import cv2
import numpy as np
from matplotlib import pyplot as plt

M = 200 # linhas
N = 400 # colunas

# IMAGEM BGR
I = np.zeros((M,N,3), np.uint8)

# Retângulo amarelo centralizado

# camada verde
I[50:150,20:380,1] = 255
# camada vermelha
I[50:150,20:380,2] = 255

cv2.imshow('Imagem colorida', I)
cv2.waitKey(0)

plt.figure()
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
plt.show()