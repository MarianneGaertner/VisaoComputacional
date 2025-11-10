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

# calcular o Score do detector de harris
R = cv2.cornerHarris(I2, 2, 3, 0.06)

# coordenadas dos pontos de interesse (corners)
limiar = 0.4*R.max()
I3 = np.uint8(R > limiar)
y, x = np.where(I3)

print(f'Número de pontos de interesse = {len(x)}')

# desenha circulos nas coordenadas dos pontos de interrese encontrados
I4 = cv2.cvtColor(I1, cv2.COLOR_GRAY2BGR)

for i in np.arange(0, len(x)):
    cv2.circle(I4, (x[i],y[i]), 10, (0,255,0), 1)

plt.figure()
plt.imshow(cv2.cvtColor(I4, cv2.COLOR_BGR2RGB))
plt.show()