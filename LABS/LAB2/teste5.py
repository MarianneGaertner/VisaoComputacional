import cv2
import numpy as np
from matplotlib import pyplot as plt

def convolucao2D(I, kernel):
    I = np.float32(I)
    lin, col, canais = I.shape
    ksize = kernel.shape[0]
    pad = ksize // 2

    # Espelha o kernel para convolução
    kernel_flipped = np.flipud(np.fliplr(kernel))

    # Saída com 3 canais
    O = np.zeros_like(I, np.float32)

    # Padding refletido
    I_padded = cv2.copyMakeBorder(I, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # Convolução canal a canal
    for c in range(canais):
        for y in range(lin):
            for x in range(col):
                regiao = I_padded[y:y+ksize, x:x+ksize, c]
                O[y, x, c] = np.sum(regiao * kernel_flipped)

    return O


def correlacao2D(I, kernel):
    I = np.float32(I)
    lin, col, canais = I.shape
    ksize = kernel.shape[0]
    pad = ksize // 2

    # saída float32
    O = np.zeros([lin, col], np.float32)

    # padding refletido para bordas
    I_padded = cv2.copyMakeBorder(I, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    for c in range(canais):
        for y in range(lin):
            for x in range(col):
                regiao = I_padded[y:y+ksize, x:x+ksize, c]
                O[y, x, c] = np.sum(regiao * kernel)
    return O


# --- Função: filtro da mediana manual ---
def filtro_mediana_manual(I_color, ksize=3):
    I_color = np.float32(I_color)
    lin, col, canais = I_color.shape
    pad = ksize // 2

    I_saida = np.zeros_like(I_color, np.float32)
    I_padded = cv2.copyMakeBorder(I_color, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    for i in range(pad, pad + lin):
        for j in range(pad, pad + col):
            janela = I_padded[i-pad:i+pad+1, j-pad:j+pad+1, :]
            I_saida[i-pad, j-pad] = np.median(janela, axis=(0, 1))
    return I_saida


# --- Leitura e preparação da imagem ---
I = cv2.imread("LABS/LAB2/Tatoo1.jpg")
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
I = np.float32(I)

# --- Etapa 1: aplicar filtro da mediana manual ---
N = 7
I_mediana = filtro_mediana_manual(I, ksize=N)

# --- Etapa 2: aplicar filtro da média (correlação) ---

kernel_media = 1/(N**2) * np.ones((N,N),np.float32)
I_mediana_media = convolucao2D(I_mediana, kernel_media)

# --- Normalização só para exibição ---
I_mostrar_orig = np.clip(I / 255, 0, 1)
I_mostrar_mediana = np.clip(I_mediana / 255, 0, 1)
I_mostrar_final = np.clip(I_mediana_media / 255, 0, 1)

# --- Exibir resultados ---
plt.figure(figsize=(14,6))

plt.subplot(1,3,1)
plt.imshow(I_mostrar_orig)
plt.title("Original (float32)")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(I_mostrar_mediana)
plt.title("Após Filtro da Mediana (5x5)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(I_mostrar_final)
plt.title("Mediana + Média (correlação 3x3)")
plt.axis("off")

plt.tight_layout()
plt.show()
