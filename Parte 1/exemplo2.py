import cv2
import numpy as np
from matplotlib import pyplot as plt

r = 30 # RAIO
x0 = 50
y0 = 50

M = 100
N = 100

I = np.zeros((M,N),np.uint8)

for theta in np.arange(0,360):
    xc = int(x0 + r*np.cos(np.deg2rad(theta)))
    yc = int(y0 - r*np.sin(np.deg2rad(theta)))
    I[xc,yc] = 255 #255->branco


plt.figure()
plt.imshow(I, cmap='gray')

# exercício 1
I2 = np.zeros((M,N),np.uint8)


for theta in np.arange(0, 360):
    for raio in np.arange(0, r):
        xc = int(x0 + raio*np.cos(np.deg2rad(theta)))
        yc = int(y0 - raio*np.sin(np.deg2rad(theta)))
        I2[xc,yc] = 255 #255->branco

plt.figure()
plt.imshow(I2, cmap='gray')
plt.show()