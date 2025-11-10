import cv2
import numpy as np
import visaoComputacional as visco
from matplotlib import pyplot as plt


I1 = cv2.imread('./Imagens/binary_dots.jpg',cv2.IMREAD_GRAYSCALE)
ret, I1 = cv2.threshold(I1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)# garantir que a imagem seja binária

plt.figure()
plt.imshow(I1,cmap='gray')
plt.title("Imagem Original")

Ifinal = visco.imclearboard(I1)

plt.figure()
plt.imshow(Ifinal,cmap='gray')
plt.title("Imagem Final")

plt.show()