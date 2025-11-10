import cv2
import numpy as np
import visaoComputacional as visco
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/binary_dots.jpg',cv2.IMREAD_GRAYSCALE)
ret, I1 = cv2.threshold(I1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)# garantir que a imagem seja binária
plt.figure()
plt.imshow(I1,cmap='gray')

# preenchimento de buracos
I_mascara = 255 - I1
plt.figure()
plt.imshow(I_mascara,cmap='gray')

I_marcador = I_mascara.copy()
I_marcador[1:-1,1:-1] = 0

plt.figure()
plt.imshow(I_marcador,cmap='gray')

I_reconstruida = visco.imreconstruction(I_mascara, I_marcador)
plt.figure()
plt.imshow(I_reconstruida,cmap='gray')

I_final = 255 - I_reconstruida
plt.figure()
plt.imshow(I_final, cmap='gray')

plt.show()