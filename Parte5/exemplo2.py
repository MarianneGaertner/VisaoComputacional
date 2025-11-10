import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/Wally.png', cv2.IMREAD_GRAYSCALE)
T1 = cv2.imread('./Imagens/template2.png', cv2.IMREAD_GRAYSCALE)


plt.figure()
plt.imshow(I1, cmap='gray')

plt.figure()
plt.imshow(T1, cmap='gray')

I_similaridade = cv2.matchTemplate(I1,T1, cv2.TM_SQDIFF)

num_linhas, num_colunas = I_similaridade.shape
y = np.arange(0, num_linhas)
x = np.arange(0, num_colunas)

X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, I_similaridade, alpha=0.5, linewidth=0.5, edgecolors='k')

limiar = 100
D = I_similaridade <= limiar
y0, x0 = np.where(D)

# desenha retângulos na imagem de entrada nas posições localizadas
I_final = I1.copy()
lin_template, col_template = T1.shape

for k in np.arange(0, x0.size):
    top_left = (x0[k], y0[k])
    bottom_right = (top_left[0] + col_template, top_left[1] + lin_template)
    cv2.rectangle(I_final, top_left, bottom_right, 0, 1)

plt.figure()
plt.imshow(I_final, cmap='gray')    

plt.show()