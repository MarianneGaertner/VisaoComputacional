import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)

# histograma
hist1 = cv2.calcHist([I1], [0], None, [256], [0, 256])
hist1 = hist1[:,0]

# pdf (função densidade acumulada)
pdf = hist1/I1.size


# função de distribuição acumulada
cdf = np.cumsum(pdf)

# função de processamento/mapeamento
f = np.uint8(256*cdf)

# equalização do histograma
n_linhas, n_colunas = I1.shape

I2 = np.zeros((n_linhas, n_colunas), np.uint8)

for x in np.arange(0, n_colunas):
    for y in np.arange(0, n_linhas):
        I2[y,x] = f[I1[y,x]] # type: ignore

# calcula o histograma da imagem em escala de cinza
hist2 = cv2.calcHist([I2], [0], None, [256], [0, 256])
hist2 = hist2[:,0]

# apresenta imagens na tela
cv2.imshow('Imagem original', I1)
cv2.imshow('Imagem processada', I2)
cv2.waitKey(0)

# apresenta histograma

plt.bar(np.arange(0,256), hist1)
plt.bar(np.arange(0,256), hist2)
plt.xlim([0,255])
plt.title('Histograma')
plt.ylabel('Contagem dos pixels')
plt.xlabel('Valores do pixels')
plt.show()
cv2.waitKey(0)