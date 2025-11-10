import cv2
import numpy as np
import visaoComputacional as visco
from matplotlib import pyplot as plt


Ifundo = cv2.imread('./Imagens/agua2.png', cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(Ifundo,cmap='gray')

Ipeixes =  cv2.imread('./Imagens/peixes2.png', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(Ipeixes,cmap='gray')

# fusão das duas imagens
alfa = 0.3
I3 = np.uint8(Ifundo*alfa + Ipeixes*(1-alfa)) 

# I3 = cv.addWeighted(Ifundo,alfa, Ipeixes, 1-alfa)

plt.figure()
plt.imshow(I3,cmap='gray')

plt.show()
