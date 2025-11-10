import cv2
import numpy as np
from matplotlib import pyplot as plt
import visaoComputacional as visco
I1 = cv2.imread('./Imagens/rice.png',cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(I1, cmap='gray')
'''
# limirização 
limiar = 110
_, Ibin1 = cv2.threshold(I1, limiar, 255, cv2.THRESH_BINARY)

plt.figure()
plt.imshow(Ibin1, cmap='gray')
'''
# Objetivo: obter a imagem que contém apenas o fundo da imagem original
# operação morfológica de erosão
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
I2 = cv2.morphologyEx(I1, cv2.MORPH_ERODE, kernel)

plt.figure()
plt.imshow(I2, cmap='gray')
# operação morfológica de dilatação
I3 = cv2.morphologyEx(I2, cv2.MORPH_DILATE, kernel)

plt.figure()
plt.imshow(I3, cmap='gray')

# 
I4 = I1 - I3
plt.figure()
plt.imshow(I4, cmap='gray')

#limiarização
limiar = 50
_, I5 = cv2.threshold(I4, limiar, 255, cv2.THRESH_BINARY)
plt.figure()
plt.imshow(I5, cmap='gray')

I6 = visco.imclearboard(I5)
plt.figure()
plt.imshow(I6, cmap='gray')


plt.show()