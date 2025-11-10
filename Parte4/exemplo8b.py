import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/moon2.jpg', cv2.IMREAD_GRAYSCALE)
I1 = np.float32(I1)/255

plt.figure()
plt.imshow(I1, cmap='gray')

# aumento de nitidez
Kl = np.array([[0,1,0], [1, -4, 1], [0,1,0]], np.float32)
Il = cv2.filter2D(I1, cv2.CV_32F, Kl, borderType=cv2.BORDER_REFLECT101)

I2 = I1 - Il
I2[I2 > 1] =1
I2[I2 < 0] = 0

plt.figure()
plt.imshow(I2, cmap='gray')
plt.show()

