import cv2
import numpy as np
from matplotlib import pyplot as plt

I = cv2.imread('./imagens/tomato_124.jpg')

plt.figure()
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))

cor_ref = np.array([32, 33, 166]) # em bgr
delta = 50

limiar_superior = cor_ref + delta
limiar_inferior = cor_ref - delta

I_bin = cv2.inRange(I, limiar_inferior, limiar_superior)

plt.figure()
plt.imshow(I_bin, cmap='gray')
plt.show()
