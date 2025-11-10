import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/pcb.jpg',cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I1, cmap='gray')

# operação morfológica de erosão
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
print(kernel)

I2 = cv2.morphologyEx(I1, cv2.MORPH_DILATE, kernel)
plt.figure()
plt.imshow(I2, cmap='gray')

plt.show()