import cv2
import numpy as np
from matplotlib import pyplot as plt

I_bgr = cv2.imread('./Imagens/flowers4.png')

cv2.imshow('Imagem BGR', I_bgr)


B = I_bgr[:,:,0]
G = I_bgr[:,:,1]
R = I_bgr[:,:,2]

# PRIMEIRA CONVERSÃO DE COLORIDO PARA ESCALA DE CINZA
I_cinza1 = np.uint8((1/3)*B + (1/3)*G +(1/3)*R)
cv2.imshow('Imagem cinza 1', I_cinza1)


# SEGUNDA CONVERSÃO DE COLORIDO PARA ESCALA DE CINZA
I_cinza2 = np.uint8(0.1140*B + 0.5870*G + 0.2989*R)
cv2.imshow('Imagem cinza 2', I_cinza2)

# TERCEIRA CONVERSÃO DE COLORIDO PARA ESCALA DE CINZA - UTILIZANDO OPENCV
I_cinza3 =cv2.cvtColor(I_bgr, cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagem cinza 3', I_cinza3)

cv2.waitKey(0)