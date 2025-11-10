import cv2
import numpy as np
import visaoComputacional as visco
from matplotlib import pyplot as plt


cap = cv2.VideoCapture('./Imagens/camera_vigilancia.avi')

n_linhas = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_colunas = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# cria buffer para armazenar os últimos N frames
tamanho_buffer = 24
buffer = visco.videoBuffer((n_linhas, n_colunas), tamanho_buffer)



while cap.isOpened():
    ret, I1 = cap.read()

    if not ret:
        print("Can’t receive frame (stream end?). Exiting ...")
        break

    # conversão de BGR para escala de cinza
    I2 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

    # insere frame atual no buffer
    buffer.insereFrame(I2.copy())

    # detecção de movimento
    I3 = np.uint8(np.abs(np.float32(buffer.primeiroFrame()) - np.float32(buffer.ultimoFrame())))

    # binarização/ limiarização
    limiar = 100
    flag, I4 = cv2.threshold(I3, limiar, 255, None, cv2.THRESH_BINARY) 


    # Apresenta frame do vídeo

    cv2.imshow('video', I4)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()