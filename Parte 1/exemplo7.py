import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/quadrados.png')
print(I1.dtype)

cv2.imshow('Imagem I1', I1)

# converter para float32
I2 = np.float32(I1/255)
print(I2.dtype)
cv2.imshow('Imagem I2', I2)

# converter novamente para uint8
I3 = np.uint8(I2*255)
cv2.imshow('Imagem I3', I3)

cv2.waitKey(0)
