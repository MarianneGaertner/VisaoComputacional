import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/marcadores.jpg')
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
I1 = cv2.resize(I1, (0,0), fx=0.25, fy=0.25)

plt.figure()
plt.imshow(I1,cmap='gray')
plt.title('Imagem Original')

# converter do tipo np.uint8 para np.float32
I2 = np.float32(I1)/255

# calcular o Score do detector de Shi-Tomazi
corners = cv2.goodFeaturesToTrack(I1, 10, 0.01, 10)
print(corners)

# desenha circulos nas coordenadas dos pontos de interrese encontrados
I4 = cv2.cvtColor(I1, cv2.COLOR_GRAY2BGR)

for corner in corners:
    print(corner[0])
    x = int(corner[0][0])
    y = int(corner[0][1])
    cv2.circle(I4, (x, y), 10, (0,255,0), 2)


plt.figure()
plt.imshow(cv2.cvtColor(I4, cv2.COLOR_BGR2RGB))
plt.show()