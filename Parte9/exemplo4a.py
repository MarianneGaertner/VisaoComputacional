import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/carta1.jpg')
I1 = cv2.resize(I1, (0,0), fx=0.25, fy=0.25)

plt.figure()
plt.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
plt.title('Imagem Original')

I2 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(I2, cmap='gray')
plt.title('Imagem em Escala de Cinza')

# detector SIFT
sift = cv2.SIFT_create()
keypoints, descritores = sift.detectAndCompute(I2, None)
print(len(keypoints))
print('1° ponto de interesse',keypoints[0].pt)
print('tamanho:', keypoints[0].size)
print('angulo', keypoints[0].angle)

print('descritores:', descritores.shape)
print(descritores[0,:])

# desenha pontos/regiões de interesse localizadas
I3 = cv2.cvtColor(I2, cv2.COLOR_GRAY2BGR)

cv2.drawKeypoints(I2, keypoints, I3, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

plt.figure()
plt.imshow(cv2.cvtColor(I3, cv2.COLOR_BGR2RGB))

plt.show()