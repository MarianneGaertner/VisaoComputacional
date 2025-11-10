import cv2
import numpy as np
from matplotlib import pyplot as plt

def canny_optimized(I, sigma, t_inferior, t_superior):
    # Converter imagem para float32 [0,1] para precisão
    I = I.astype(np.float32) / 255.0
    lin, col = I.shape

    # ------------------------
    # PASSO 1: Suavização Gaussiana
    # ------------------------
    w = int(6*sigma + 1)
    kernel = np.zeros((w, w), np.float32)
    frac_k = 1 / (2 * np.pi * sigma**2)
    frac_exp = -1 / (2 * sigma**2)
    for y in range(w):
        for x in range(w):
            j = y - (w-1)/2
            i = x - (w-1)/2
            kernel[y,x] = frac_k * np.exp(frac_exp*(i**2 + j**2))
    kernel /= kernel.sum()

    Is = cv2.filter2D(I, -1, kernel, borderType=cv2.BORDER_REFLECT101)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(I, cmap='gray')
    plt.title("Imagem Original")
    plt.subplot(1,2,2)
    plt.imshow(Is, cmap='gray')
    plt.title("Imagem Suavizada")

    # ------------------------
    # PASSO 2: Gradientes
    # ------------------------
    Kx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]], np.float32)
    Ky = Kx.T
    dx = cv2.filter2D(Is, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)
    dy = cv2.filter2D(Is, cv2.CV_32F, Ky, borderType=cv2.BORDER_REFLECT101)

    mag = np.sqrt(dx**2 + dy**2)         
    alfa = np.rad2deg(np.arctan2(dy, dx)) 
    alfa[alfa < 0] += 180                

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(mag*255, cmap='gray')
    plt.title("Magnitude do gradiente")
    plt.subplot(1,2,2)
    plt.imshow(alfa*255, cmap='hsv')
    plt.title("Direção do gradiente (α)")

    # ------------------------
    # PASSO 3: Supressão de Não Máximos
    # ------------------------
    Gn = np.zeros([lin, col], np.float32)
    for i in range(1, lin-1):
        for j in range(1, col-1):
            ang = alfa[i,j]
            # Determinar direção discreta
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                viz1, viz2 = mag[i, j-1], mag[i, j+1]   # horizontal
            elif 22.5 <= ang < 67.5:
                viz1, viz2 = mag[i-1, j+1], mag[i+1, j-1] # diagonal /
            elif 67.5 <= ang < 112.5:
                viz1, viz2 = mag[i-1, j], mag[i+1, j]   # vertical
            else:
                viz1, viz2 = mag[i-1, j-1], mag[i+1, j+1] # diagonal \

            if mag[i,j] >= viz1 and mag[i,j] >= viz2:
                Gn[i,j] = mag[i,j]

    plt.figure()
    plt.imshow(Gn, cmap='gray')
    plt.title("Supressão de Não Máximos (Gn)")

    # ------------------------
    # PASSO 4: Limiarização Dupla
    # ------------------------
    # Conversão para uint8 apenas nas máscaras
    GNH = (Gn >= t_superior).astype(np.uint8) * 255  # bordas fortes
    GNL = (Gn >= t_inferior).astype(np.uint8) * 255  # bordas fracas + fortes
    GNL2 = GNL - GNH  # bordas fracas isoladas

    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    plt.imshow(GNH, cmap='gray')
    plt.title("GNH - bordas fortes")
    plt.subplot(1,3,2)
    plt.imshow(GNL, cmap='gray')
    plt.title("GNL - bordas fracas + fortes")
    plt.subplot(1,3,3)
    plt.imshow(GNL2, cmap='gray')
    plt.title("GNL - bordas fracas isoladas")

    # ------------------------
    # PASSO 5: Histerese (conectividade)
    # ------------------------
    Ib = np.copy(GNH)
    for v in range(1, lin-1):
        for u in range(1, col-1):
            if GNH[v,u] > 0:
                for i in range(-1,2):
                    for j in range(-1,2):
                        if GNL2[v+i, u+j] > 0:
                            Ib[v+i, u+j] = 255
    # Garantir que bordas fortes permaneçam
    Ib = cv2.add(Ib, GNH)

    return Ib.astype(np.uint8)  # saída final como uint8

# ------------------------
# Teste
# ------------------------
I = cv2.imread("LABS/LAB2/castle.jpg", cv2.IMREAD_GRAYSCALE)
w = 3
sigma = (w-1)/6
t_superior = 0.9
t_inferior = 0.1

IB = canny_optimized(I, sigma, t_inferior, t_superior)

plt.figure()
plt.imshow(IB, cmap='gray')
plt.title("Canny Optimizado - Saída Final")
plt.show()
