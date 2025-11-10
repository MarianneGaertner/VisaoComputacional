import cv2
import numpy as np
import visaoComputacional as visco
from matplotlib import pyplot as plt


I = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

# limiarização global
L = 170
O = visco.limiarizacao_global2(I, L)

plt.figure()
plt.imshow(O, cmap='gray')
plt.show()