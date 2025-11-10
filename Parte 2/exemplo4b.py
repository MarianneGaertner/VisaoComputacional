import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/capa.jpg')

plt.figure()
plt.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))

# Homografia planar

# pontos de origem
pts_src = np.array([[52, 100], [396, 79], [86, 571], [448, 527]])

# pontos de destino
n_linhas = 350
n_colunas = 250 

pts_dst = np.array([[0, 0],[n_colunas, 0],[0, n_linhas],[n_colunas, n_linhas]])

H, _ = cv2.findHomography(pts_src, pts_dst)

I2 = cv2.warpPerspective(I1, H, (n_colunas, n_linhas))

plt.figure()
plt.imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))
plt.show()
