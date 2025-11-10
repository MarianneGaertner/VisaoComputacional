import cv2          # importa biblioteca opencv
import numpy as np  # biblioteca numpy para processamento de arranjos e matrizes  

# ----------------------------------------------------------------------------
# A cada clique do mouse sobre a imagem é apresentado as coordenadas (y,x)
# do pixel clicado e o seu valor (cor) em BGR, RGB e HSV
# ----------------------------------------------------------------------------

# funcao de callback
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('I[',y,',', x,'] = ')
        cor_bgr = I[y,x].reshape((1,1,3))
        cor_rgb = cv2.cvtColor(cor_bgr, cv2.COLOR_BGR2RGB)
        cor_hsv = cv2.cvtColor(cor_bgr, cv2.COLOR_BGR2HSV)

        print('BGR: ', cor_bgr)
        print('RGB: ', cor_rgb)
        print('HSV: ', cor_hsv)
        print('\n')

# ----------------------------------------------------------------------------
# lê imagem de um arquivo
I = cv2.imread('./Imagens/tomato_124.jpg')

# apresenta imagem na tela
cv2.imshow('image', I)

# define função click_event como função de callback do mouse
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
# ----------------------------------------------------------------------------