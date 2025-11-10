import cv2
import numpy as np
from matplotlib import pyplot as plt

I = cv2.imread('./imagens/tomato_124.jpg')

plt.figure()
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))



# segmentação baseada em cor
r_ref = 166  # referencia vermelha
g_ref = 32 # referencia verde
b_ref = 33 # referencia azul

delta = 50

B = I[:,:,0]
G = I[:,:,1]
R = I[:,:,2]

Mb = (B >= b_ref - delta) & (B <= b_ref + delta)
Mg = (G >= g_ref - delta) & (G <= g_ref + delta)
Mr = (R >= r_ref - delta) & (R <= r_ref + delta)

M = Mb & Mg & Mr
n_linhas, n_colunas, n_camadas = I.shape

I_bin = np.zeros((n_linhas, n_colunas), np.uint8)

I_bin[M] = 255

plt.figure()
plt.imshow(I_bin, cmap='gray')
plt.show()
