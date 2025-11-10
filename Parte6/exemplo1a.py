import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/rice.png', cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(I1, cmap='gray')

# limiarização global
limiar = 128
ret, I2 = cv2.threshold(I1, limiar, 255, cv2.THRESH_BINARY)

plt.figure()
plt.imshow(I2, cmap='gray')

# operação de erosão
S = np.ones((2,2), np.uint8)
#I3 = cv2.erode(I2, S, iterations=1)
I3 = cv2.morphologyEx(I2, cv2.MORPH_ERODE, S)

plt.figure()
plt.imshow(I3, cmap='gray')

plt.show()