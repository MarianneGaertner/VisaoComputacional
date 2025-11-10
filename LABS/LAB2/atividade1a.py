import numpy as np
import cv2
from matplotlib import pyplot as plt

# -------------------------------------------------------------------------
# 1️⃣ Filtro da mediana manual na área
def filtro_mediana_area(I, x, y, largura, altura, tamanho_janela):
    m = tamanho_janela // 2
    lin, col, canais = I.shape
    O = np.copy(I)

    for c in range(canais):
        for yy in range(y + m, min(y + altura - m, lin - m)):
            for xx in range(x + m, min(x + largura - m, col - m)):
                regiao = I[yy - m:yy + m + 1, xx - m:xx + m + 1, c]
                valores = np.sort(regiao.flatten())
                mediana = valores[len(valores)//2]
                O[yy, xx, c] = mediana
    return O

# -------------------------------------------------------------------------
# 2️⃣ Filtro de média manual na área (passa-baixa)
def filtro_media_area(I, x, y, largura, altura, tamanho_janela):
    m = tamanho_janela // 2
    lin, col, canais = I.shape
    O = np.copy(I)

    for c in range(canais):
        for yy in range(y + m, min(y + altura - m, lin - m)):
            for xx in range(x + m, min(x + largura - m, col - m)):
                regiao = I[yy - m:yy + m + 1, xx - m:xx + m + 1, c]
                O[yy, xx, c] = np.mean(regiao)
    return O

# -------------------------------------------------------------------------
# 3️⃣ Determina a cor da pele ignorando pixels escuros
def cor_pele_area(I, x, y, largura, altura, limiar=0.3):
    regiao = I[y:y+altura, x:x+largura, :]
    media_lum = np.mean(regiao, axis=2)
    mascara_pele = media_lum > limiar
    if np.sum(mascara_pele) == 0:
        return np.mean(regiao, axis=(0,1))
    cor_pele = np.mean(regiao[mascara_pele], axis=0)
    return cor_pele

# -------------------------------------------------------------------------
# 4️⃣ Filtro adaptativo puxando pixels para cor da pele
def filtro_puxa_pele(I, x, y, largura, altura, cor_pele, alpha=0.5):
    O = np.copy(I)
    for yy in range(y, y+altura):
        for xx in range(x, x+largura):
            cor_pixel = I[yy, xx, :]
            dist = np.linalg.norm(cor_pixel - cor_pele)
            peso = np.exp(-dist**2 / (2*0.1**2))
            O[yy, xx, :] = (1 - alpha*peso)*cor_pixel + alpha*peso*cor_pele
    return O

# -------------------------------------------------------------------------
# Leitura e normalização da imagem
img = cv2.imread("LABS/LAB2/Tatoo1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.float32(img) / 255

# Retângulo da tatuagem
x, y, largura, altura = 229, 82, 212, 480

# Parâmetros dos filtros
tam_mediana = 7
tam_media = 5
alpha_cor = 0.6

# -------------------------------------------------------------------------
# 1️⃣ Aplicar mediana
img_med = filtro_mediana_area(img, x, y, largura, altura, tam_mediana)

# 2️⃣ Aplicar média / suavização
img_suave = filtro_media_area(img_med, x, y, largura, altura, tam_media)

# 3️⃣ Determinar cor da pele
cor_pele = cor_pele_area(img_suave, x, y, largura, altura, limiar=0.3)
print("Cor da pele estimada:", cor_pele)

# 4️⃣ Aplicar filtro adaptativo puxando para a cor da pele
img_final = filtro_puxa_pele(img_suave, x, y, largura, altura, cor_pele, alpha=alpha_cor)

# -------------------------------------------------------------------------
# Exibição dos resultados
plt.figure(figsize=(18,5))

plt.subplot(1,4,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(img_med)
plt.title("Mediana")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(img_suave)
plt.title("Mediana + Média")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(np.clip(img_final,0,1))
plt.title("Mediana + Média + Cor da Pele")
plt.axis("off")

plt.tight_layout()

# Retângulo da área
img_ret = np.copy(img)
cv2.rectangle(img_ret, (x, y), (x + largura, y + altura), (1, 0, 0), 2)
plt.figure()
plt.imshow(img_ret)
plt.title("Região filtrada (em vermelho)")
plt.axis("off")
plt.show()
