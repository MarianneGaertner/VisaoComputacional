import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I1, cmap='gray')

t_lower = 50
t_upper = 150
I__borda = cv2.Canny(I1,t_lower,t_upper)
plt.figure()
plt.imshow(I__borda, cmap='gray')



plt.show()

