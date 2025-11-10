import cv2
import numpy as np
from matplotlib import pyplot as plt
import visaoComputacional as visco

I1 = cv2.imread('./Imagens/tomato_124.jpg')
plt.figure()
plt.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))

# segmentação baseada em cor
cor_ref = np.array([27, 32, 177])
limiar = 75
I2 = visco.color_segmentation(I1, cor_ref, limiar)

plt.figure()
plt.imshow(I2, cmap='gray')

# operação morfológica de fechamento
S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 12))
I3 = cv2.morphologyEx(I2, cv2.MORPH_CLOSE, S)

plt.figure()
plt.imshow(I3, cmap='gray')

# operação morfológica de abertura
S2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
I4 = cv2.erode(I3, S2)
I5 = cv2.dilate(I4, S2)

plt.figure()
plt.imshow(I5,cmap='gray')
plt.show()
