import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I1, cmap='gray')

# Derivada em relação a x
Kx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]], np.float32)
print(Kx)

Ix = cv2.filter2D(I1, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)
print(Ix)
plt.figure()
plt.imshow(Ix, cmap='gray')
# Derivada em relação a y
Ky = np.transpose(Kx)
Iy = cv2.filter2D(I1, cv2.CV_32F, Ky, borderType=cv2.BORDER_REFLECT101)
plt.figure()
plt.imshow(Iy, cmap='gray')

# Imagem da magnitude do gradiente
M = np.sqrt(Ix**2 + Iy**2)
plt.figure()
plt.imshow(M, cmap='gray')
print(np.max(M))

# limiarização global
limiar = 300
_, Iborda = cv2.threshold(M, limiar, 255, cv2.THRESH_BINARY)

plt.figure()
plt.imshow(Iborda, cmap='gray')


plt.show()