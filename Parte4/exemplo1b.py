import cv2
import numpy as np
from matplotlib import pyplot as plt

# IMAGEM SINTÉTICA
I1 = np.zeros((5,8), np.uint8)
I1[:,5:] = 255

print(I1)

# KERNEL 
K = np.ones((3,3), np.float32)
print(K)

# Filtragem 2D

I2 = cv2.filter2D(I1, cv2.CV_32F, K, borderType=cv2.BORDER_REFLECT101)
print(I2)
