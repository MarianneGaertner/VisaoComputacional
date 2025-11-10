import cv2
import numpy as np
from matplotlib import pyplot as plt

I_gray = cv2.imread("./Imagens/castle.jpg", cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(I_gray, cmap='gray')

# histograma
hist = cv2.calcHist([I_gray], [0], None, [256], [0, 256])
plt.bar(np.arange(0,256), hist[:,0])
plt.xlim([0, 256])
plt.title('Histograma')

plt.show
