import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/monalisa2.png', cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(I1, cmap='gray')

# SUAVIZAÇÃO
# kernel normalizado
w = 5
K = 1/(w**2) *  np.ones((w,w), np.float32)
print(K)

# filtragem 2D
I2 = cv2.filter2D(I1, -1, K, borderType=cv2.BORDER_REFLECT101)
plt.figure()
plt.imshow(I2, cmap='gray')
plt.show()