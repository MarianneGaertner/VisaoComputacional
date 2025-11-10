import cv2
import numpy as np
from matplotlib import pyplot as plt


I = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

# limiarização global
L = 170

ret, O = cv2.threshold(I, L, 255, None, cv2.THRESH_BINARY)

plt.figure()
plt.imshow(O, cmap='gray')
plt.show()
