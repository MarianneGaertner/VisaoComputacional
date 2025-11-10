import cv2
import numpy as np
from matplotlib import pyplot as plt


def canny(I, sigma, t_inferior,t_superior):
    lin, col = I.shape
       
    # PASSO 1
    w = int(6*sigma+1)
    
    kernel = np.zeros((w,w), np.float32)

    frac_k = 1/(2* np.pi *(sigma**2))
    frac_exp = - 1/(2*(sigma**2))

    for y in np.arange(0, w):
        for x in np.arange(0, w):
            j = y - (w-1)/2
            i = x - (w-1)/2
            kernel[y,x] = frac_k*np.exp(frac_exp*(j**2 + i**2))

    kernel = kernel/np.sum(kernel)

    Is = cv2.filter2D(I, -1, kernel, borderType=cv2.BORDER_REFLECT101)
    
    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(I, cmap='gray')
    # plt.title("a) Passo 1 - Imagem Original")
    # plt.subplot(1, 2, 2)
    # plt.imshow(Is, cmap='gray')
    # plt.title("b) Passo 1 - Imagem Suavizada")

    # PASSO 2 - Computar a magnitude M e a fase alfa do gradiente de Is
    
    Kx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.transpose(Kx)
    dx = cv2.filter2D(Is, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)
    dy = cv2.filter2D(Is, cv2.CV_32F, Ky, borderType=cv2.BORDER_REFLECT101)
    
    mag = np.sqrt(dx**2 + dy**2)
        
    alfa = np.rad2deg(np.arctan2(dy,dx))
     

    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(mag, cmap='gray')
    # plt.title("Passo 2 - a) Magnitude (M)")
    # plt.subplot(1, 2, 2)
    # plt.imshow(alfa, cmap='hsv')
    # plt.title("Passo 2 - b) Fase (α em graus)")


    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.title("a) Gradiente Gx")
    # plt.imshow(dx, cmap='gray')

    # plt.subplot(1, 2, 2)
    # plt.title("b) Gradiente Gy")
    # plt.imshow(dy, cmap='gray')

    
    # PASSO 3 - Supressão de não máximos locais
    
    Gn = np.zeros([lin, col], np.float32)
    alfa[alfa < 0] += 180
    mag = (mag/(mag.max()+1e-10))* 255

    for i in range(1, lin-1):
        for j in range(1, col-1):
            
            ang = alfa[i, j]
            
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                direcao = 0
            elif (22.5 <= ang < 67.5):
                direcao = 45
            elif (67.5 <= ang < 112.5):
                direcao = 90
            else:
                direcao = 135

            # Compara vizinhos
            if direcao == 0:
                viz1, viz2 = mag[i, j-1], mag[i, j+1]
            elif direcao == 45:
                viz1, viz2 = mag[i-1, j-1], mag[i+1, j+1]
            elif direcao == 90:
                viz1, viz2 = mag[i-1, j], mag[i+1, j]
            else:
                viz1, viz2 = mag[i+1, j-1], mag[i-1, j+1]

            if mag[i, j] >= viz1 and mag[i, j] >= viz2:
                Gn[i, j] = mag[i, j]

    
    '''
    plt.figure()
    plt.imshow(Gn, cmap='gray')
    plt.title("Passo 3 - Supressão de Não Máximos (Gn)")
    '''
    # PASSO 4 - Processo de limiarização duplo
    GNH = np.zeros([lin,col], np.uint8)
    GNL = np.zeros([lin, col], np.uint8)
    
    GNH = ((Gn >= t_superior)*255).astype(np.uint8)
    GNL = ((Gn >= t_inferior)*255).astype(np.uint8)
    
    '''
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(GNH, cmap='gray')
    plt.title("Passo 4 - a) GNH")
    plt.subplot(1, 2, 2)
    plt.imshow(GNL, cmap='gray')
    plt.title("Passo 4 - b) GNL")
    ''' 
    GNL2 = GNL-GNH
    '''
    plt.figure()
    plt.imshow(GNL2, cmap='gray')
    plt.title("Passo 4 - GNL Atualizado ")
    '''
    
    # PASSO 5 - Análise de conectividade
    Ib = np.copy(GNH)
    linhas, colunas = Ib.shape
    for v in range(1, linhas-1):
        for u in range(1, colunas-1):
            if(GNH[v,u]>0):
                for i in range(-1,2):
                    for j in range(-1,2):
                        if GNL2[v+i, u+j] > 0:
                            Ib[v+i, u+j] = 255                   
    Ib = Ib + GNH
    
    
    return Ib
'''
I = cv2.imread("LABS/LAB2/castle.jpg", cv2.IMREAD_GRAYSCALE)
w = 5
sigma = (w-1)/6

t_superior = 90 #110  80  70  100 55
t_inferior = 30 #55  30  25 20 26


IB = canny(I, sigma, t_inferior, t_superior)


plt.figure()
plt.imshow(IB, cmap='gray')
plt.title("Imagem de Borda")

plt.figure()
plt.imshow(cv2.Canny(I, t_inferior, t_superior), cmap='gray')
plt.title("Função OpenCv")

plt.show()
'''