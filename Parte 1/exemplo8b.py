import cv2
import numpy as np
from matplotlib import pyplot as plt

I_bgr = cv2.imread('./Imagens/quadrados.png')

plt.figure()
plt.imshow(cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB))
plt.show()

# conversão BGR para HSV
I_hsv = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2HSV)

print(f'[100:100] = {I_hsv[100, 100]}')
print(f'[100:300] = {I_hsv[100, 300]}')
print(f'[300:100] = {I_hsv[300, 100]}')
print(f'[300:300] = {I_hsv[300, 300]}')