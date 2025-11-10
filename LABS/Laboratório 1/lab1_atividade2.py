#import visaoComputacional as visco
import cv2
import numpy as np

# abrir arquivos de videos
#cap1 = cv2.VideoCapture('Chromakey.mp4')
#cap2 = cv2.VideoCapture('Clouds.mp4')
cap1 = cv2.VideoCapture(r'C:\Users\maria\OneDrive\Área de Trabalho\Marianne\UFSC\9° FASE\Visão Computacional em Robótica\Materiais de Aula\Laboratórios\Laboratório 1\Chromakey.mp4')
cap2 = cv2.VideoCapture(r'C:\Users\maria\OneDrive\Área de Trabalho\Marianne\UFSC\9° FASE\Visão Computacional em Robótica\Materiais de Aula\Laboratórios\Laboratório 1\Clouds.mp4')
n_linhas = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_colunas  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video = cv2.VideoWriter('resultado_atividade2.mp4', fourcc, 24, (n_colunas, n_linhas))


M1 = np.zeros((n_linhas, n_colunas), dtype=np.uint8)  # fundo
M2 = np.zeros((n_linhas, n_colunas), dtype=np.uint8) 

#RGB   78, 212, 190

b_ref = 190
g_ref = 212
r_ref = 78
tau = 41

# processa cada quadro dos vídeos
ret = True
while ret:

    # lê frame dos vídeos
    ret, I1 = cap1.read()
    _, I2 = cap2.read()

    if I1 is None:
        break
    
    I2 = cv2.resize(I2, (n_colunas, n_linhas))
    
    # Efeito de chromakey
    B = I1[:,:,0]
    G = I1[:,:,1]
    R = I1[:,:,2]

    diff_B = np.float32(B)-b_ref
    diff_G = np.float32(G)-g_ref
    diff_R = np.float32(R)-r_ref

    D = (np.sqrt(diff_B**2 + diff_G**2 + diff_R**2))

    M1 = 255*(np.ones((n_linhas, n_colunas), np.uint8))
    indices = D <= tau
    M1[indices] = 0 
    M2  = cv2.bitwise_not(M1)  
        
    I_frontal = cv2.bitwise_and(I1,I1,mask=M1)
    I_fundo   = cv2.bitwise_and(I2,I2,mask=M2)   
    I_final = cv2.add(I_frontal, I_fundo)

    cv2.imshow("Chromakey", I_final)
    video.write(I_final)
    



     # "PLOTAR"
    new_size = (n_colunas//2, n_linhas//2)
    I1_small = cv2.resize(I1, new_size)
    I2_small = cv2.resize(I2, new_size)
    I_final_small = cv2.resize(I_final, new_size)
  
    
    # converte máscaras para 3 canais
    M1_small = cv2.cvtColor(cv2.resize(M1, new_size), cv2.COLOR_GRAY2BGR)
    M2_small = cv2.cvtColor(cv2.resize(M2, new_size), cv2.COLOR_GRAY2BGR)

    # organiza em grid 2x2
    top = np.hstack((I1_small, I2_small))
    bottom = np.hstack((M1_small, M2_small))
    grid = np.vstack((top, bottom))
    cv2.imshow("Chromakey Grid 2x2", grid)
    
    if cv2.waitKey(5) == ord('q'):
        break

cv2.destroyAllWindows()
video.release()