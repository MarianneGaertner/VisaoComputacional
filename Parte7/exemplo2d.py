import cv2
import numpy as np
from matplotlib import pyplot as plt

# Funções ---------------------------
def bounding_box(I1):
    y, x = np.where(I1)
    ymin = np.min(y)
    ymax = np.max(y)
    xmin = np.min(x)
    xmax = np.max(x)

    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])

    return p0, p1

def mpq(I1, p, q):
    y, x = np.where(I1)

    mpq = np.float32(np.sum((x**p)*(y**q)))

    return mpq
# -----------------------------------------------------

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
p0, p1 = bounding_box(I_componente)


# desenha bounding box
I2 = cv2.cvtColor(I_componente, cv2.COLOR_GRAY2BGR)

color = (255, 128, 128)
thickness = 1
cv2.rectangle(I2, p0, p1, color, thickness)


# cálculo dos momentos
m00 = mpq(I_componente, 0,0)
m10 = mpq(I_componente, 1, 0)
m01 = mpq(I_componente, 0, 1)

print(f'Momento m00 = {m00}\n')
print(f'Momento m10 = {m10}\n')
print(f'Momento m01 = {m01}\n')

# centroide
xc = m10/m00
yc = m01/m00

centroide = np.array([xc, yc])
print(centroide)

# desenha centroide
thickness = -1
radius = 2
cv2.circle(I2, np.int32(centroide), radius, color, thickness)


plt.figure()
plt.imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))
plt.show()