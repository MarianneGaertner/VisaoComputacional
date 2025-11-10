import cv2
import numpy as np
from matplotlib import pyplot as plt

I1 = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)

# altera¸c~ao de contraste
alfa = 1.1
I2 = cv2.multiply(I1, alfa)


# apresenta imagens na tela
cv2.imshow('Imagem original', I1)
cv2.imshow('Imagem processada', I2)
cv2.waitKey(0)


# histograma
hist1 = cv2.calcHist([I1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([I2],[0], None, [256], [0,256])

# apresenta histograma
plt.bar(np.arange(0,256), hist1[:,0])
plt.bar(np.arange(0,256), hist2[:,0])
plt.xlim([0,255])
plt.title('Histograma')
plt.ylabel('Contagem dos pixels')
plt.xlabel('Valores do pixels')
plt.show()
cv2.waitKey(0)
