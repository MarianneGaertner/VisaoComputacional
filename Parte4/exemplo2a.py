import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(I1,cmap='gray')


# kernel
K = np.ones((3,3),np.float32)

I2 = cv2.filter2D(I1, -1, K, borderType=cv2.BORDER_REFLECT101)
plt.figure()
plt.imshow(I2, cmap='gray')
plt.show()