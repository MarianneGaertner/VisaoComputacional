import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
I1 = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)

# histograma
hist1 = cv2.calcHist([I1], [0], None, [256], [0, 256])

hist1 = hist1[:, 0]

# Equalização histograma

I2 = cv2.equalizeHist(I1)

hist2 = cv2.calcHist([I2], [0], None, [256], [0, 256])
hist2 = hist2[:, 0]

# apresenta imagens na tela
cv2.imshow('Imagem original', I1)
cv2.imshow('Imagem processada', I2)

# apresenta histograma
plt.bar(np.arange(0,256), hist1)
plt.bar(np.arange(0,256), hist2)
plt.xlim([0,255])
plt.title('Histograma')
plt.ylabel('Contagem dos pixels')
plt.xlabel('Valores do pixels')
plt.show()
cv2.waitKey(0)