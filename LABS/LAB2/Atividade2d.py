import cv2
import numpy as np
from matplotlib import pyplot as plt

def canny(I, sigma, t_inferior, t_superior):
    # Converter imagem para float32 no intervalo [0,1]
    I = I.astype(np.float32) / 255.0
    lin, col = I.shape
    print("Tipo imagem:", I.dtype, "| Intervalo:", I.min(), I.max())
    
   
    w = int(6 * sigma + 1)
    kernel = np.zeros((w, w), np.float32)

    frac_k = 1 / (2 * np.pi * sigma**2)
    frac_exp = -1 / (2 * sigma**2)

    for y in range(w):
        for x in range(w):
            j = y - (w - 1) / 2
            i = x - (w - 1) / 2
            kernel[y, x] = frac_k * np.exp(frac_exp * (j**2 + i**2))
    kernel /= kernel.sum()

    Is = cv2.filter2D(I, -1, kernel, borderType=cv2.BORDER_REFLECT101)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap='gray')
    plt.title("Passo 1 - Imagem Original")
    plt.subplot(1, 2, 2)
    plt.imshow(Is, cmap='gray')
    plt.title("Passo 1 - Imagem Suavizada")

   
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], np.float32)
    Ky = Kx.T

    dx = cv2.filter2D(Is, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)
    dy = cv2.filter2D(Is, cv2.CV_32F, Ky, borderType=cv2.BORDER_REFLECT101)

    mag = np.sqrt(dx**2 + dy**2)
    

    alfa = np.rad2deg(np.arctan2(dy, dx))
    

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mag, cmap='gray')
    plt.title("Passo 2 - Magnitude (M)")
    plt.subplot(1, 2, 2)
    plt.imshow(alfa, cmap='hsv')
    plt.title("Passo 2 - Direção (α em graus)")

    # -----------------------------------------
    # PASSO 3 – Supressão de não máximos locais
    # -----------------------------------------
    Gn = np.zeros([lin, col], np.float32)
    alfa[alfa < 0] += 180
    mag /= mag.max() + 1e-10   # normalizar para [0,1]

    for i in range(1, lin - 1):
        for j in range(1, col - 1):
            ang = alfa[i, j]

            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                direcao = 0
            elif (22.5 <= ang < 67.5):
                direcao = 45
            elif (67.5 <= ang < 112.5):
                direcao = 90
            else:
                direcao = 135

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

    plt.figure()
    plt.imshow(Gn, cmap='gray')
    plt.title("Passo 3 - Supressão de Não Máximos (Gn)")

    # -----------------------------------
    # PASSO 4 – Limiarização dupla [0,1]
    # -----------------------------------
    GNH = (Gn >= t_superior).astype(np.float32)*1
    GNL = (Gn >= t_inferior).astype(np.float32)*1
    GNL2 = GNL - GNH

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(GNH, cmap='gray')
    plt.title("GNH - bordas fortes")
    plt.subplot(1, 3, 2)
    plt.imshow(GNL, cmap='gray')
    plt.title("GNL - bordas fracas + fortes")
    plt.subplot(1, 3, 3)
    plt.imshow(GNL2, cmap='gray')
    plt.title("GNL - bordas fracas isoladas")

    # -------------------------------
    # PASSO 5 – Histerese (conectividade)
    # -------------------------------
    Ib = np.copy(GNH)
    for v in range(1, lin - 1):
        for u in range(1, col - 1):
            if GNH[v, u] > 0:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if GNL[v + i, u + j] > 0:
                            Ib[v + i, u + j] = 1.0
    Ib[Ib!=0] = 1.0

    print("Tipo Ib:", Ib.dtype, "| Intervalo:", Ib.min(), Ib.max())
    return Ib

# -------------------------------
# TESTE
# -------------------------------
I = cv2.imread("LABS/LAB2/castle.jpg", cv2.IMREAD_GRAYSCALE)
w = 3
sigma = (w-1)/6

#  0 - 1
t_superior = 0.25
t_inferior = 0.07

IB = canny(I, sigma, t_inferior, t_superior)

plt.figure()
plt.imshow(IB, cmap='gray')
plt.title("Ib - Canny (0–1)")

plt.figure()
plt.imshow(cv2.Canny(I, int(t_inferior*255), int(t_superior*255)), cmap='gray')
plt.title("Função OpenCV (referência)")
plt.show()
