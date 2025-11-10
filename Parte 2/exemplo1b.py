import cv2
import numpy as np
from matplotlib import pyplot as plt

I = cv2.imread('./Imagens/monalisa2.png', cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(I, cmap='gray')

# escalamento
sx = 0.5
sy = 1.25

I2 = cv2.resize(I, None, fx=sx, fy=sy)

plt.figure()
plt.imshow(I2, cmap='gray')

plt.show()
