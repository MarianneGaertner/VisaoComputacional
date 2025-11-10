import cv2
import numpy as np
from matplotlib import pyplot as plt


def canny(I, sigma, t_inferior, t_superior):
    lin, col = I.shape

    # =========================================================
    # PASSO 1 — SUAVIZAÇÃO GAUSSIANA
    # =========================================================
    w = int(6 * sigma + 1)
    kernel = np.zeros((w, w), np.float32)

    frac_k = 1 / (2 * np.pi * (sigma ** 2))
    frac_exp = -1 / (2 * (sigma ** 2))

    for y in range(w):
        for x in range(w):
            j = y - (w - 1) / 2
            i = x - (w - 1) / 2
            kernel[y, x] = frac_k * np.exp(frac_exp * (j ** 2 + i ** 2))

    kernel /= np.sum(kernel)
    Is = cv2.filter2D(I, -1, kernel, borderType=cv2.BORDER_REFLECT101)

    # Exibição Passo 1
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap='gray')
    plt.title("Passo 1 - Imagem Original")
    plt.subplot(1, 2, 2)
    plt.imshow(Is, cmap='gray')
    plt.title("Passo 1 - Imagem Suavizada")

    # =========================================================
    # PASSO 2 — MAGNITUDE E DIREÇÃO DO GRADIENTE
    # =========================================================
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], np.float32)
    Ky = np.transpose(Kx)

    dx = cv2.filter2D(Is, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)
    dy = cv2.filter2D(Is, cv2.CV_32F, Ky, borderType=cv2.BORDER_REFLECT101)

    mag = np.sqrt(dx ** 2 + dy ** 2)
    mag = np.clip(mag, 0, np.percentile(mag, 99))  # corta outliers
    mag = (mag / (mag.max() + 1e-10)) * 255

    alfa = np.rad2deg(np.arctan2(dy, dx))
     

    # Exibição Passo 2
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im1 = axs[0].imshow(mag, cmap='gray')
    axs[0].set_title("(a) Magnitude (M)")
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(alfa, cmap='hsv')
    axs[1].set_title("(b) Direção (α em graus)")
    fig.colorbar(im2, ax=axs[1], label="Ângulo (°)")
    plt.tight_layout()

    # =========================================================
    # PASSO 3 — SUPRESSÃO DE NÃO MÁXIMOS (quantizado)
    # =========================================================
    Gn = np.zeros((lin, col), np.float32)
    alfa[alfa < 0] += 180 
    
    for i in range(1, lin - 1):
        for j in range(1, col - 1):
            ang = alfa[i, j]

            # Quantiza o ângulo em 4 direções
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
                viz1, viz2 = mag[i, j - 1], mag[i, j + 1]
            elif direcao == 45:
                viz1, viz2 = mag[i - 1, j + 1], mag[i + 1, j - 1]
            elif direcao == 90:
                viz1, viz2 = mag[i - 1, j], mag[i + 1, j]
            else:
                viz1, viz2 = mag[i - 1, j - 1], mag[i + 1, j + 1]

            if mag[i, j] >= viz1 and mag[i, j] >= viz2:
                Gn[i, j] = mag[i, j]

    # Exibição Passo 3
    plt.figure()
    plt.imshow(Gn, cmap='gray')
    plt.title("Passo 3 - Supressão de Não Máximos (Gn)")

    # =========================================================
    # AJUSTE DE CONTRASTE E HISTOGRAMA
    # =========================================================
    Gn = Gn * 1.5
    Gn = np.clip(Gn, 0, 255)

    plt.figure()
    plt.hist(Gn.ravel(), bins=50, color='gray')
    plt.title("Histograma da Magnitude após Supressão")
    plt.xlabel("Intensidade")
    plt.ylabel("Frequência")

    # =========================================================
    # PASSO 4 — LIMIARIZAÇÃO DUPLA (relativa)
    # =========================================================
    Gn_norm = Gn / (np.max(Gn) + 1e-10)
    GNH = (Gn_norm >= t_superior).astype(np.float32) * 255
    GNL = (Gn_norm >= t_inferior).astype(np.float32) * 255
    GNL = GNL - GNH

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(GNH, cmap='gray')
    plt.title("Passo 4 - GNH (Fortes)")
    plt.subplot(1, 2, 2)
    plt.imshow(GNL, cmap='gray')
    plt.title("Passo 4 - GNL (Fracos)")

    # =========================================================
    # PASSO 5 — ANÁLISE DE CONECTIVIDADE
    # =========================================================
    Ib = np.copy(GNH)
    for i in range(1, lin - 1):
        for j in range(1, col - 1):
            if GNH[i, j] > 0:
                viz = GNL[i - 1:i + 2, j - 1:j + 2]
                Ib[i - 1:i + 2, j - 1:j + 2] = np.maximum(Ib[i - 1:i + 2, j - 1:j + 2], viz)

    Ib = np.clip(Ib, 0, 255).astype(np.uint8)

    plt.figure()
    plt.imshow(Ib, cmap='gray')
    plt.title("Passo 5 - Ib (Resultado Final)")

    return Ib


# =========================================================
# EXECUÇÃO
# =========================================================
I = cv2.imread("LABS/LAB2/castle.jpg", cv2.IMREAD_GRAYSCALE)
w = 3
sigma = (w - 1) / 6

# Limiar relativo (0–1)
t_superior = .9
t_inferior = 0.1

IB = canny(I, sigma, t_inferior, t_superior)

# Comparação com OpenCV
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(IB, cmap='gray')
plt.title("Implementação Manual (Ib)")

plt.subplot(1, 2, 2)
plt.imshow(cv2.Canny(I, 70, 25), cmap='gray')
plt.title("Função cv2.Canny()")
plt.tight_layout()
plt.show()
