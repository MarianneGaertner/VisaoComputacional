import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/formas.png',cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(I1,cmap='gray')

# rotulamento
num_elem, I_labels = cv2.connectedComponents(I1)
print(num_elem)
print(I_labels.shape)
print(np.max(I_labels))
print(np.min(I_labels))

I2 = np.uint8(np.float32(I_labels)/num_elem * 255)
plt.figure()
plt.imshow(I2, cmap='gray')
plt.show()