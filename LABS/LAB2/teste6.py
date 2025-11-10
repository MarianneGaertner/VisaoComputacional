import numpy as np
import cv2
from matplotlib import pyplot as plt

# -------------------------------------------------------------------------
# Correlação 2D genérica
def correlacao2D(I, kernel):
    lin, col, canais = I.shape
    O = np.zeros(I.shape, np.float32)
    m = kernel.shape[0] // 2

    for c in range(canais):
        for y in range(m, lin - m):
            for x in range(m, col - m):
                regiao = I[y - m:y + m + 1, x - m:x + m + 1, c]
                O[y, x, c] = np.sum(regiao * kernel)
    return O

# -------------------------------------------------------------------------
# Filtro da mediana manual — apenas dentro do retângulo
def filtro_mediana_area(I, x, y, largura, altura, tamanho_janela):
    m = tamanho_janela // 2
    lin, col, canais = I.shape
    O = np.copy(I)

    for c in range(canais):
        for yy in range(y + m, min(y + altura - m, lin - m)):
            for xx in range(x + m, min(x + largura - m, col - m)):
                regiao = I[yy - m:yy + m + 1, xx - m:xx + m + 1, c]
                # Ordena e pega o valor do meio (sem usar np.median)
                valores = np.sort(regiao.flatten())
                mediana = valores[len(valores) // 2]
                O[yy, xx, c] = mediana
    return O

# -------------------------------------------------------------------------
# Filtro da média manual — apenas dentro do retângulo
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
# Filtro gaussiano simples (opcional, pode substituir a média)
def kernel_gaussiano(N, sigma):
    ax = np.arange(-(N//2), N//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return np.float32(kernel)
def filtro_bilateral_area(I, x, y, largura, altura, tamanho_janela, sigma_espaco, sigma_cor):
    """
    I: imagem RGB float32 [0..1]
    x, y, largura, altura: região do retângulo a suavizar
    tamanho_janela: tamanho da vizinhança (ímpar)
    sigma_espaco: controla a influência da distância espacial (quanto maior, mais ampla a suavização)
    sigma_cor: controla a influência da diferença de cor (quanto maior, mais "mistura" tons diferentes)
    """
    m = tamanho_janela // 2
    lin, col, canais = I.shape
    O = np.copy(I)

    # Pré-calcula o peso espacial (depende só da distância)
    ax = np.arange(-m, m + 1)
    xx, yy = np.meshgrid(ax, ax)
    peso_espacial = np.exp(-(xx**2 + yy**2) / (2 * sigma_espaco**2))

    for c in range(canais):
        for yy in range(y + m, min(y + altura - m, lin - m)):
            for xx in range(x + m, min(x + largura - m, col - m)):
                regiao = I[yy - m:yy + m + 1, xx - m:xx + m + 1, c]

                # Diferença de cor entre o pixel central e os vizinhos
                diff_cor = regiao - I[yy, xx, c]

                # Peso de cor (quanto mais parecida a cor, mais pesa)
                peso_cor = np.exp(-(diff_cor**2) / (2 * sigma_cor**2))

                # Combinação dos pesos (espacial × cor)
                pesos = peso_espacial * peso_cor

                # Normalização dos pesos
                pesos /= np.sum(pesos)

                # Média ponderada
                O[yy, xx, c] = np.sum(regiao * pesos)

    return O
# -------------------------------------------------------------------------
# Leitura e normalização da imagem
img = cv2.imread("LABS/LAB2/Tatoo1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.float32(img) / 255

# Coordenadas da área da tatuagem
x, y, largura, altura = 229, 82, 212, 480

# Parâmetros dos filtros
tam_mediana = 9
tam_media = 7
janela = 7
# -------------------------------------------------------------------------
# 1️⃣ Aplicar filtro da mediana (somente na área)
img_med = filtro_mediana_area(img, x, y, largura, altura, tam_mediana)

# 2️⃣ Aplicar filtro da média (somente na mesma área)
img_med_media = filtro_media_area(img_med, x, y, largura, altura, tam_media)

# 3️⃣ (opcional) aplicar um segundo passo de média para suavizar ainda mais
img_med_media2 = filtro_media_area(img_med_media, x, y, largura, altura, tam_media)



tam_janela = 15
sigma_espaco = 8
sigma_cor = 0.2

# Aplicação
img_bilateral = filtro_bilateral_area(img, x, y, largura, altura, janela, sigma_espaco, sigma_cor)

# -------------------------------------------------------------------------
# Exibição dos resultados
plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(img_med)
plt.title("Após Filtro da Mediana")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(img_med_media)
plt.title("Mediana + 1x Média")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(img_med_media2)
plt.title("Mediana + 2x Média (suavizado)")
plt.axis("off")

plt.tight_layout()

# -------------------------------------------------------------------------
# Mostra a região filtrada
img_ret = np.copy(img)
cv2.rectangle(img_ret, (x, y), (x + largura, y + altura), (1, 0, 0), 2)
plt.figure()
plt.imshow(img_ret)
plt.title("Região suavizada (em vermelho)")
plt.axis("off")




plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(np.clip(img_bilateral, 0, 1))
plt.title("Filtro Bilateral (apenas tatuagem)")
plt.axis("off")

plt.tight_layout()

# Mostra o retângulo delimitado
img_ret = np.copy(img)
cv2.rectangle(img_ret, (x, y), (x + largura, y + altura), (1, 0, 0), 2)
plt.figure()
plt.imshow(img_ret)
plt.title("Região suavizada (em vermelho)")
plt.axis("off")
plt.show()