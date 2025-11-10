import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread("./Imagens/castle.jpg", cv2.IMREAD_GRAYSCALE)


# histograma
hist1 = cv2.calcHist([I1], [0], None, [256], [0, 256])

# ALTERAÇÕES DE BRILHO
alfa = 80

n_linhas, n_colunas = I1.shape
I2 = np.zeros((n_linhas, n_colunas), np.uint8)

for x in np.arange(0, n_colunas):
    for y in np.arange(0, n_linhas):
        if (int(I1[y, x]) + alfa > 255):
            I2[y, x] = I1[y, x] + alfa
        elif (int(I1[y, x]) + alfa < 0):
            I2[y, x] = 0
        else:
            I2[y, x] = np.uint8(I1[y, x] + alfa)


hist2 = cv2.calcHist([I2], [0], None, [256], [0, 256])

fig, axs = plt.subplots(2, 2, figsize=(8, 8))

axs[0, 0].imshow(I1, cmap='gray')
axs[0, 1].bar(np.arange(0, 256), hist1[:, 0])
axs[1, 0].imshow(I2, cmap='gray')
axs[1, 1].bar(np.arange(0, 256), hist2[:, 0])
plt.show()
