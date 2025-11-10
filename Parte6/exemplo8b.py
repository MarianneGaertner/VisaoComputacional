import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/pcb.jpg',cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I1, cmap='gray')
plt.title('Imagem Original')
# operação morfológica de fechamento
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#EROSÃO
I2 = cv2.morphologyEx(I1, cv2.MORPH_ERODE, kernel)
plt.figure()
plt.imshow(I2, cmap='gray')
plt.title("Imagem Pós Erosão")

# DILATAÇÃO
I3 = cv2.morphologyEx(I2, cv2.MORPH_DILATE, kernel)
plt.figure()
plt.imshow(I3, cmap='gray')
plt.title("Imagem Dilatada")


plt.show()