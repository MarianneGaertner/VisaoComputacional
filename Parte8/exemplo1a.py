import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/capa.jpg')

plt.figure()
plt.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
plt.title('Imagem Original')

I2 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(I2, cmap='gray')
plt.title('Imagem em escala de cinza')

# imagem de borda
I3 = cv2.Canny(I2, 50, 150, apertureSize=3)
plt.figure()
plt.imshow(I3, cmap='gray')
plt.title('Imagem de Borda')

# extração de contornos
contours, hierachy = cv2.findContours(I3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

I4 = np.zeros(I3.shape, np.uint8)
color = (255, 255, 255)

for i in range(len(contours)):
    if len(contours[i][:,0,0]) > 100:
        cv2.drawContours(I4, contours, i, color)

lines = cv2.HoughLines(I4, rho=1, theta=np.pi/180, threshold=120)
print(lines) # [r, theta]
print(lines.shape)

I5 = cv2.cvtColor(I4, cv2.COLOR_GRAY2BGR)
for r_theta in lines:
    r = r_theta[0][0] 
    theta = r_theta[0][1]
    
    x0 = r*np.cos(theta)
    y0 = r*np.sin(theta)

    x1 = int(x0 - 1000*np.sin(theta))
    y1 = int(y0 + 1000*np.cos(theta))

    x2 = int(x0 + 1000*np.sin(theta))
    y2 = int(y0 - 1000*np.cos(theta))

    cv2.line(I5, (x1,y1), (x2,y2), (0,0,255), 2)
    

plt.figure()
plt.imshow(cv2.cvtColor(I5, cv2.COLOR_BGR2RGB))
plt.show()