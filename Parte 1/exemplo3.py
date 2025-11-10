import cv2
import numpy as np
from matplotlib import pyplot as plt

N = 400
M = 200

I = np.zeros((M,N), np.uint8)

for x in np.arange(0, N):
    c = np.uint8(x*255/(N-1))
    I[:,x] = c


plt.figure()
plt.imshow(I, cmap='gray')


# exercício 2
n = 100
m = 100

r = 40 # RAIO
x0 = 50
y0 = 50

I2 = np.zeros((m, n),np.uint8)


for y in range(m):
    for x in range(n):
        dist = np.sqrt((y-y0)**2 + (x-x0)**2)
        if dist <= r:
            I2[x,y] = np.uint8(255*(1-dist/r))
        else:
            I2[x,y] = 0
        
plt.figure()
plt.imshow(I2, cmap='gray')
plt.show()
    