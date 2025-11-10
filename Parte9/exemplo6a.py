import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/carta1.jpg')
I1 = cv2.resize(I1, (0,0), fx=0.25, fy=0.25)
plt.figure()
plt.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
plt.title('Imagem Original')
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(I1, cmap='gray')
plt.title('Imagem em Escala de Cinza')

I2 = cv2.imread('./Imagens/cena2.jpg')
I2 = cv2.resize(I2, (0,0), fx=0.2, fy=0.2)
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(I2, cmap='gray')

# detector SIFT
sift = cv2.SIFT_create()
kp1, descritores1 = sift.detectAndCompute(I1, None)
kp2, descritores2 = sift.detectAndCompute(I2, None)

# cria objeto BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# realiza o match entre os descritores da image I1 e da imagem I2
matches = bf.match(descritores1, descritores2)
print(len(matches))
print(matches[0])
print(matches[0].queryIdx) # indice do ponto de interesse da imagem
print(matches[0].trainIdx) # indice do ponto de interesse da outra imagem
print(matches[0].distance)

# ordena os matches pela menor distância
matches = sorted(matches, key=lambda x:x.distance)
print(matches[0].queryIdx) 
print(matches[0].trainIdx) 
print(matches[0].distance)

print('\n')
print(matches[1].queryIdx) 
print(matches[1].trainIdx) 
print(matches[1].distance)

# visualizar de forma gráfica
I3 = cv2.drawMatches(I1, kp1, I2, kp2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure()
plt.imshow(cv2.cvtColor(I3, cv2.COLOR_BGR2RGB))

# obter matriz de homografia que mapeia a imagem I1 na cena da imagem I2
N = 100
src_pts = np.zeros((100, 1, 2), np.float32)
dst_pts = np.zeros((100, 1, 2), np.float32)

for i in np.arange(0, N):
    src_pts[i] = np.float32(kp1[matches[i].queryIdx].pt).reshape(-1, 1, 2)
    dst_pts[i] = np.float32(kp2[matches[i].trainIdx].pt).reshape(-1, 1, 2)

# matriz de homografia
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

# localização da imagem I1 na imagem I2
h, w = I1.shape
pts = np.float32([[0,0],[0, h-1], [w-1, h-1], [w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, M)

I4 = cv2.cvtColor(I2, cv2.COLOR_GRAY2BGR)
I4 = cv2.polylines(I4, [np.int32(dst)], True, (0,255,0),2, cv2.LINE_AA)

plt.figure()
plt.imshow(cv2.cvtColor(I4, cv2.COLOR_BGR2RGB))
plt.show()