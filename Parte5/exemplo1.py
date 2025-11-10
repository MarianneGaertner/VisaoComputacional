import cv2
import numpy as np
from matplotlib import pyplot as plt

def similaridade(I1, I2):
    I1 = np.float32(I1)/255
    I2 = np.float32(I2)/255

    valor_similaridade = np.sum(np.abs(I1 - I2))

    return valor_similaridade

#Imagem de busca
I1 = cv2.imread('./Imagens/Wally.png', cv2.IMREAD_GRAYSCALE)
#template
T1 = cv2.imread('./Imagens/template2.png', cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(I1, cmap='gray')

plt.figure()
plt.imshow(T1, cmap='gray')


lin_I1,  col_I1 = I1.shape
lin_template,  col_template = T1.shape

n_linhas = lin_I1 - lin_template +1
n_colunas = col_I1 - col_template +1

I_similaridade = np.zeros((n_linhas, n_colunas), np.float32)

for x in np.arange(0, col_I1 - col_template + 1):
    for y in np.arange(0, lin_I1-lin_template+1):

        janela = I1[y:y+lin_template, x:x+col_template] 

        I_similaridade[y,x] = similaridade(janela, T1)

# apresenta imagem de siliaridade como superfície
y = np.arange(0, n_linhas)
x = np.arange(0, n_colunas)

X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, I_similaridade, alpha=0.5, linewidth=0.5, edgecolors='k')

# localizar o template na imagem de entrada
limiar = 100
#limiar = 1.1*np.min(I_similaridade)
D = I_similaridade < limiar

y0, x0 = np.where(D)

# desenha retângulos na imagem de entrada nas posições localizadas
I_final = I1.copy()

for k in np.arange(0, x0.size):
    top_left = (x0[k], y0[k])
    bottom_right = (top_left[0] + col_template, top_left[1] + lin_template)
    cv2.rectangle(I_final, top_left, bottom_right, 0, 1)

plt.figure()
plt.imshow(I_final, cmap='gray')    

plt.show()

