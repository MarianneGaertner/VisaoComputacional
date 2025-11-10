import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/cell.png', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I1, cmap='gray')

# limiarização global
limiar = 128
_, I2 = cv2.threshold(I1,limiar, 255, cv2.THRESH_BINARY)
plt.figure()
plt.imshow(I2,cmap='gray')

# imagem de borda
Kx = np.array([[-1, 0,1], [-2, 0, 2], [-1, 0, 1]], np.float32)
Ky = np.transpose(Kx)

Ix = cv2.filter2D(I1, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)
Iy = cv2.filter2D(I1, cv2.CV_32F, Ky, borderType=cv2.BORDER_REFLECT101)
M = np.sqrt(Ix**2 + Iy**2)
print(np.max(M))

limiar_borda = 100
_, I_BORDA = cv2.threshold(M, limiar_borda, 255, cv2.THRESH_BINARY)
plt.figure()
plt.imshow(I_BORDA,cmap='gray')
plt.show()