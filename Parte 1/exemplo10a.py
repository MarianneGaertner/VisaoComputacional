import cv2
import numpy as np
from matplotlib import pyplot as plt

I = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')


# Histograma
hist = np.zeros(256)
n_linhas, n_colunas = I.shape

for x in np.arange(0, n_colunas):
    for y in np.arange(0, n_linhas):
        hist[I[y, x]] = hist[I[y, x]] + 1

 # apresenta histograma
plt.figure()
plt.bar(np.arange(0, 256), hist)
plt.xlim([0, 256])
plt.title('Histograma')
plt.ylabel('Contagem dos pixels')
plt.xlabel('Valores do pixels')

# função do opencv para calcular o histograma
hist2 = cv2.calcHist([I], [0], None, [256], [0, 256])
plt.figure()
plt.bar(np.arange(0, 256), hist2[:, 0])
plt.xlim([0, 256])
plt.title('Histograma - função opencv')
plt.ylabel('Contagem dos pixels')
plt.xlabel('Valores do pixels')

plt.show()
