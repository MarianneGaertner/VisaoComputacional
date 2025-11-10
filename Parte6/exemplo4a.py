import cv2
import numpy as np
from matplotlib import pyplot as plt

def im_reconstructuion(I_mascara, I_marcador, S):
    num_pixels = 0
    while num_pixels != np.sum(I_marcador):
        num_pixels = np.sum(I_marcador)
        I_marcador = cv2.dilate(I_marcador, S)
        I_marcador = cv2.bitwise_and(I_marcador, I_mascara)
    
    return I_marcador

I1 = cv2.imread('./Imagens/castle.jpg',cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I1,cmap='gray')

# limiarização global
limiar = 128
ret, I2 = cv2.threshold(I1, limiar, 255, cv2.THRESH_BINARY)
plt.figure()
plt.imshow(I2,cmap='gray')

# reconstrução morfológica 
I_marcador = np.zeros(I2.shape, np.uint8)
I_marcador[637, 336] = 255

I_mascara = I2.copy()
S = np.ones((3,3), np.float32)

I_final = im_reconstructuion(I_mascara, I_marcador, S)

plt.figure()
plt.imshow(I_final, cmap='gray')
plt.show()