import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/moon2.jpg', cv2.IMREAD_GRAYSCALE)
I1 = np.float32(I1)/255

plt.figure()
plt.imshow(I1, cmap='gray')

# aumento de nitidez
w = 11
sigma = (w-1)/6
Is = cv2.GaussianBlur(I1, (w,w), sigma)
plt.figure()
plt.imshow(Is, cmap='gray')

Ig = I1 - Is
plt.figure()
plt.imshow(Ig, cmap='gray')

k = 10
I2 = I1+k*Ig
I2[I2>1] = 1
I2[I2<0] = 0
plt.figure()
plt.imshow(I2, cmap='gray')

plt.show()

