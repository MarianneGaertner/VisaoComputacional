import cv2
import numpy as np
import visaoComputacional as visco
from matplotlib import pyplot as plt

def detectaBordas(I1):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ix = cv2.filter2D(I1, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)

    Ky = np.transpose(Kx)
    Iy = cv2.filter2D(I1, cv2.CV_32F, Ky, borderType=cv2.BORDER_REFLECT101)

    M = np.sqrt(Ix**2 + Iy**2)
    M = cv2.convertScaleAbs(M)  # converte float -> uint8

    limiar = 150
    _, borda = cv2.threshold(M, limiar, 255, cv2.THRESH_BINARY)
    return borda

tempo_inicio = cv2.getTickCount()

# Leitura da imagem
I1 = cv2.imread('Tatoo1.jpg')
Igray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

# ROI
x1, y1, x2, y2 = 240, 80, 435, 566
I_display = np.copy(I1)
cv2.rectangle(I_display, (x1, y1), (x2, y2), (0,0,0))

# Detectar bordas em tons de cinza
borda = detectaBordas(Igray)

# Limiarização para achar partes pretas
limiar = 144
_, pretoInterno = cv2.threshold(Igray, limiar, 255, cv2.THRESH_BINARY_INV)

# Combina os dois resultados
final = cv2.bitwise_or(borda, pretoInterno)

O = I1.copy()

# percorre regiao de interesse procurando pixels em branco na imagem final. Quando encontrar faz a media dos pixels ao redor na imagem original e adiciona na imagem O no lugar
for linha in np.arange(y1+1, y2-1):
    for coluna in np.arange(x1+1, x2-1):
        if final[linha, coluna] == 255:
            # --- tamanho da vizinhança ---
            tamanho_kernel = 33   
            raio = tamanho_kernel // 2

            # Monta o kernel NxN de vizinhança centrado no pixel atual
            kernel = np.ones((tamanho_kernel, tamanho_kernel, 3), np.float32)
            for y in range(-raio, raio + 1):
                for x in range(-raio, raio + 1):
                    for camada in range(3):
                        if I1[linha + y, coluna + x, camada] == 0:
                            continue
                        kernel[y + raio, x + raio, camada] = I1[linha + y, coluna + x, camada]
            
            # calcula a média espacial separada por canal (R,G,B) 
            media_por_canal = np.mean(kernel, axis=(0,1)) 

            # garante tipo uint8 antes de escrever 
            media_por_canal = np.round(media_por_canal).astype(np.uint8) 

            # atribui os 3 canais no pixel de saída
            O[linha, coluna, :] = media_por_canal
w = 9
sigma = (w-1)/6

K = visco.gaussianKernel(w, sigma)
O[y1:y2,x1:x2,:] = cv2.filter2D(O[y1:y2,x1:x2,:], -1, K, borderType=cv2.BORDER_REFLECT101)

tempo_final = cv2.getTickCount()

fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
ax1.set_title('Original')

ax2 = fig.add_subplot(2,2,2)
ax2.imshow(borda, cmap='gray')
ax2.set_title('Deteccao de Borda')

ax3 = fig.add_subplot(2,2,3)
ax3.imshow(pretoInterno, cmap='gray')
ax3.set_title('Limiarizacao (partes pretas)')

ax4 = fig.add_subplot(2,2,4)
ax4.imshow(final, cmap='gray')
ax4.set_title('Deteccao final (bordas + preto)')

plt.tight_layout() 

fig2 = plt.figure()
ax1 = fig2.add_subplot(1,2,1)
ax1.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
ax1.set_title('Original')

ax2 = fig2.add_subplot(1,2,2)
ax2.imshow(cv2.cvtColor(O, cv2.COLOR_BGR2RGB))
ax2.set_title('Resultado')
tempo_de_processamento = (tempo_final - tempo_inicio)/cv2.getTickFrequency()

print(f"Processamento: {tempo_de_processamento}")
plt.show()

