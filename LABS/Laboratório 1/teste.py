import cv2
import numpy as np

# abrir arquivos de videos
cap1 = cv2.VideoCapture(r'C:\Users\maria\OneDrive\Área de Trabalho\Marianne\UFSC\9° FASE\Visão Computacional em Robótica\Materiais de Aula\Laboratórios\Laboratório 1\Chromakey.mp4')
cap2 = cv2.VideoCapture(r'C:\Users\maria\OneDrive\Área de Trabalho\Marianne\UFSC\9° FASE\Visão Computacional em Robótica\Materiais de Aula\Laboratórios\Laboratório 1\Clouds.mp4')

n_linhas = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_colunas = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))

# saída em MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('resultado_chromakey.mp4', fourcc, 24, (n_colunas, n_linhas))

# cor de referência (formato BGR!)
b_ref, g_ref, r_ref = 189, 211, 77
tau = 60  # tolerância

while True:
    ret1, I1 = cap1.read()
    ret2, I2 = cap2.read()

    if not ret1 or not ret2:
        break

    # separa canais BGR
    B, G, R = I1[:,:,0], I1[:,:,1], I1[:,:,2]

    # distância euclidiana entre pixel e cor de referência
    diff_B = np.int16(B) - b_ref
    diff_G = np.int16(G) - g_ref
    diff_R = np.int16(R) - r_ref
    D = np.sqrt(diff_B**2 + diff_G**2 + diff_R**2)

    # gera máscaras binárias em uint8
    M1 = np.zeros((n_linhas, n_colunas), dtype=np.uint8)  # fundo
    M2 = np.zeros((n_linhas, n_colunas), dtype=np.uint8)  # frente

    M1[D <= tau] = 255   # região verde → fundo
    M2[D > tau]  = 255   # resto → frente

    # expande para 3 canais
    M1_rgb = np.repeat(M1[:, :, np.newaxis], 3, axis=2)
    M2_rgb = np.repeat(M2[:, :, np.newaxis], 3, axis=2)

    # aplica chromakey só com uint8
    I_frontal = (I1 * (M2_rgb // 255)).astype(np.uint8)
    I_fundo   = (I2 * (M1_rgb // 255)).astype(np.uint8)

    I_final = cv2.add(I_frontal, I_fundo)

    # mostra e salva
    cv2.imshow("Chromakey", I_final)
    video.write(I_final)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
video.release()
cv2.destroyAllWindows()
