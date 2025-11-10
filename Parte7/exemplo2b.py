import cv2
import numpy as np
from matplotlib import pyplot as plt


I1 = cv2.imread('./Imagens/formas.png',cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(I1,cmap='gray')

# rotulamento
num_elem, I_labels = cv2.connectedComponents(I1)

# seleção de regiões
rotulo_desejado = 1
I_componente = np.uint8((I_labels == rotulo_desejado))*255
plt.figure()
plt.imshow(I_componente, cmap='gray')

# bounding box
y, x = np.where(I_componente)

ymin = np.min(y)
ymax = np.max(y)
xmin = np.min(x)
xmax = np.max(x)

p0 = np.array([xmin, ymin])
p1 = np.array([xmax, ymax])

print(p0)
print(p1)

# desenha bounding box
I2 = cv2.cvtColor(I_componente, cv2.COLOR_GRAY2BGR)

color = (255, 128, 128)
thickness = 1
cv2.rectangle(I2, p0, p1, color, thickness)

plt.figure()
plt.imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))
plt.show()