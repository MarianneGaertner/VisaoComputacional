import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('/Imagens/greenscreen.jpg')

plt.figure()
plt.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))

# segmentação baseada em cor/geração de máscara
cor_ref = np.array([99, 212, 168])# BGR
delta = 40

M1 = cv2.inRange(I1, cor_ref - delta, cor_ref + delta)

#obtenção de negativo/ geração de segunda máscara
M2 = 255 - M1 

plt.figure()
plt.imshow(M1, cmap='gray')

# operação lógica AND entre I1 e a máscara M2
I2 = cv2.bitwise_and(I1, I1, mask=M2)
plt.figure()
plt.imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))

plt.show()


