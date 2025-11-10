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
plt.show()