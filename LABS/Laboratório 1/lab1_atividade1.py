import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# ----------------------------------------------------------
#cap = cv2.VideoCapture('sequencia1.mp4')
#cap = cv2.VideoCapture(r'C:\Users\maria\OneDrive\Área de Trabalho\Marianne\UFSC\9° FASE\Visão Computacional em Robótica\Materiais de Aula\Laboratórios\Laboratório 1\sequencia1.mp4')
cap = cv2.VideoCapture(r'C:\Users\maria\OneDrive\Área de Trabalho\Marianne\UFSC\9° FASE\Visão Computacional em Robótica\Materiais de Aula\Laboratórios\Laboratório 1\Atividade1.mp4')
n_linhas = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_colunas  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# ----------------------------------------------------------
# Gera figura
fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=0.9)

fig_width = 20
fig_height = 6
fig.set_size_inches(fig_width/2.54, fig_height/2.54)
fig.tight_layout()

# ----------------------------------------------------------
ret = True
# cor_ref rgb(208,227,95) ->BGR(95,227,208)
# Seção de variáveis
#cor_ref = np.array([95,227,208])# BGR

# ref pilha rgb = 35,83,189
b_ref = 189 # ref pilha
g_ref = 83 # ref pilha
r_ref = 35 # ref pilha
'''
b_ref = 95 # ref bola
g_ref = 223 # ref bola
r_ref = 204 # ref bola'''
qtd = 0
tau = 90 #limiar
M = np.zeros((n_linhas, n_colunas), np.uint8)
presenca_anterior = False
t_total = 0
qtd_frames = 0
print("coluna:")

print(n_colunas)
print(n_colunas//2)

while ret:
    tempo1 = cv2.getTickCount()
    # lê frame do vídeo
    ret, I1 = cap.read()

    if I1 is None:
        break
    
    # Algoritmo de detecção de objetos
    B = I1[:,:,0]
    G = I1[:,:,1]
    R = I1[:,:,2]

    diff_B = np.float32(B)-b_ref
    diff_G = np.float32(G)-g_ref
    diff_R = np.float32(R)-r_ref

    D = np.uint8(np.sqrt(diff_B**2 + diff_G**2 + diff_R**2))
    #D = uint8(np.sqrt((B-b_ref)**2 + (G-g_ref)**2 + (R-r_ref)**2))
    
    M[:] = 0 
    indices = (D <= tau)
    M[indices] = 255

    coluna = M[:, n_colunas//2]
    
    presenca_atual = False  
    for y in range(n_linhas):
        if (coluna[y] == 255):
            presenca_atual = True
            break
   
    if presenca_atual and not presenca_anterior:
        qtd += 1

    presenca_anterior = presenca_atual

    tempo2 = cv2.getTickCount()
    t_frame = (tempo2-tempo1)/cv2.getTickFrequency()

    t_total += t_frame
    qtd_frames += 1

    print("Qtd:", qtd)
    

    # atualiza plot
    ax1.clear()
    ax1.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
    ax1.set_title("Frame original")

    ax2.clear()
    ax2.imshow(M, cmap='gray')
    ax2.plot([n_colunas/2, n_colunas/2], [0, n_linhas-1], ':')
    ax2.set_title("Máscara binária")
    
    
    ax3.clear()
    ax3.plot(np.arange(0, coluna.size), coluna)
    ax3.set_ylim(0, 260)
    ax3.set_xlim(0, coluna.size)
    ax3.set_title('Coluna central da\n imagem binária')
    plt.pause(0.05)   
    

print('Qtd contabilizada: ', qtd)
print('Processo finalizado!')
print('Tempo médio por quadro: {:.4f} segundos'.format(t_total / qtd_frames))
plt.show()
cap.release()
cv2.waitKey(0)