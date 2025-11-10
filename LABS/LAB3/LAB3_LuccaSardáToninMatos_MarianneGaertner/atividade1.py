import cv2
import numpy as np
import VisaoComputacional as visco
from matplotlib import pyplot as plt


# ALFABETO - template -----------------------------------------------------------------------
I_letras = cv2.imread('LABS/LAB3/banco_de_imagens/template_letras.png', cv2.IMREAD_GRAYSCALE)
_, I_bin_letras = cv2.threshold(I_letras, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# plt.figure()
# plt.imshow(I_letras, cmap='gray')
#plt.title('Imagem Original Alfabeto')

# plt.figure()
# plt.imshow(I_bin_letras, cmap='gray')
#plt.title('Imagem alfabeto binarizado')


infoLetras = visco.analisaRegioes(I_bin_letras)


# ordena as letras pelo centroide
letras = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T', 'U','V','X','Z','W']
centroides = []
for info in infoLetras:
    centroides.append(info['centroide'][0])

indices_ordenados = np.argsort(centroides)

templates = {}
for i in range(len(letras)):
    idx = indices_ordenados[i]
    letra = letras[i]
    templates[letra] = infoLetras[idx]['imagem']

# Modifica o tamanho das letras para tamanho fixo (altura_base x largura_base), igual à primeira letra 
x1, y1 = infoLetras[0]['bb_p1']
x0, y0 = infoLetras[0]['bb_p0']
altura_base = y1-y0
largura_base = x1-x0

templates_norm = visco.modifica_tamanho_img(templates, altura_base, largura_base)


# plt.figure(figsize=(15, 3))
# i = 1
# for letra, img in templates_norm.items():
#     plt.subplot(3, 9, i)  
#     plt.imshow(img, cmap='gray')
#     plt.title(letra)
#     plt.axis('off')
#     i += 1
# plt.tight_layout()


# ETIQUETAS --------------------------------------------------------------------------------
I1 = cv2.imread('LABS/LAB3/banco_de_imagens/im1.png')
# plt.figure()
# plt.imshow(I1)
# plt.title("Etiqueta Original")

# Segmentação
n_linhas, n_colunas, n_camadas = I1.shape
ref_color = np.array([99, 169, 113], np.uint8)
limiar = 50
Ref = np.zeros((n_linhas, n_colunas, n_camadas), np.float32)
Ref[:,:,0] = ref_color[0]
Ref[:,:,1] = ref_color[1]
Ref[:,:,2] = ref_color[2]

D = np.sqrt(np.sum((np.float32(I1) - Ref)**2, axis=2))

I_bin = np.zeros((n_linhas, n_colunas), np.uint8)
I_bin[D <= limiar] = 255

# plt.figure()
# plt.imshow(I_bin,cmap='gray')
# plt.title('Etiqueta Binarizada')

# analisar regiões
infoRegioes = visco.analisaRegioes(I_bin)


num_elem, I_labels = cv2.connectedComponents(I_bin)
centroides = np.zeros((num_elem - 1, 2), np.float32)
circularidade = np.zeros(num_elem - 1, np.float32)

for n in np.arange(0, num_elem-1):
    circularidade[n] = infoRegioes[n]['circularidade']
    centroides[n] = infoRegioes[n]['centroide']
#print('Centroides = ',centroides)
#print('Circularidade = ', circularidade)

index_quadrado = np.argmin(circularidade)
coordenada_quadrado = centroides[index_quadrado]
infoQuadrado = infoRegioes[index_quadrado]
#print(coordenada_quadrado)

centroides_organizados = [coordenada_quadrado]

outros_centroides = np.delete(centroides, index_quadrado, axis=0)

# Superior direito
index_superior_dir = np.argmin(outros_centroides[:,1])
superior_direito =  outros_centroides[index_superior_dir]
centroides_organizados.append(superior_direito)
#print('superior_direita', superior_direito)
outros_centroides = np.delete(outros_centroides, index_superior_dir, axis=0)

# Inferior esquerdo
index_inferior_esq = np.argmin(outros_centroides[:,0])
inferior_esquerdo = outros_centroides[index_inferior_esq]
centroides_organizados.append(inferior_esquerdo)
outros_centroides = np.delete(outros_centroides, index_inferior_esq, axis=0)
#print("outros centroidesssss", outros_centroides)

index_inferior_dir= np.argmax(outros_centroides[:,0]) 
inferior_direito = outros_centroides[index_inferior_dir]
centroides_organizados.append(inferior_direito)

centroides_organizados = np.array(centroides_organizados, np.float32)
#print('centroides orgaizados', centroides_organizados)


# I_vis = I1.copy()
# for i, c in enumerate(centroides_organizados):
#     x, y = np.uint32(c)
#     cv2.circle(I_vis, (x, y), 7, (0, 0, 255), -1)
#     cv2.putText(I_vis, str(i), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# plt.figure()
# plt.imshow(cv2.cvtColor(I_vis, cv2.COLOR_BGR2RGB))
# plt.title("Centróides Numerados")

# Homografia planar ------------------------------------------------------------------------------
# pontos de origem 
pts_src = centroides_organizados
# pontos destino
n_linhas = 320
n_colunas = 762 

pts_dst = np.array([[0, 0],[n_colunas, 0],[0, n_linhas],[n_colunas, n_linhas]])

H, _ = cv2.findHomography(pts_src, pts_dst)

I2 = cv2.warpPerspective(I1, H, (n_colunas, n_linhas))

# plt.figure()
# plt.imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))
# plt.title('Etiqueta Alinhada')

margem = 20
I3 = I2[margem:-margem, margem:-margem]
# plt.figure()
# plt.imshow(I3)
# plt.title('Etiqueta com Bordas Recortadas')

I_gray = cv2.cvtColor(I3, cv2.COLOR_RGB2GRAY) 

# fazer com que a imagem seja binária
_, I_bin_txt = cv2.threshold(I_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 

# plt.figure()
# plt.imshow(I_bin_txt, cmap='gray')
# plt.title('Etiqueta Binarizada')

# Identificar linhas
linhas = visco.identificar_linhas(I_bin_txt, kernel_linha=(70,3))

#print(f"Total de palavras extraídas: {len(linhas)}")
# ---------------------------------------

# Informações das linhas----------------------------------------
kernel_linhas = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))

# Linha 1
I_dilatada_1 = cv2.dilate(linhas[0], kernel_linhas, iterations=1)
# plt.figure()
# plt.imshow(I_dilatada_1, cmap='gray') 
# plt.title("Dilatação Horizontal - Linha 1")
palavras_1 = visco.identificar_palavras_linha(I_dilatada_1,linhas[0])
letrasLinha1 = visco.identificar_letras(palavras_1)
letrasLinha1_norm = visco.normaliza_letras_linha(letrasLinha1, altura_base, largura_base)

linha1_reconhecida = []
for palavra in letrasLinha1_norm:
    texto = visco.reconhece_palavra(palavra, templates_norm, metrica='ZNCC')
    linha1_reconhecida.append(texto)
print('Linhas Reconhecidas:')
print( *linha1_reconhecida)

# Linha 2 
I_dilatada_2 = cv2.dilate(linhas[1], kernel_linhas, iterations=1)
# plt.figure()
# plt.imshow(I_dilatada_2, cmap='gray') 
# plt.title("Dilatação Horizontal - Linha 2")
palavras_2 = visco.identificar_palavras_linha(I_dilatada_2, linhas[1])
letrasLinha2 = visco.identificar_letras(palavras_2)

letrasLinha2_norm = visco.normaliza_letras_linha(letrasLinha2, altura_base, largura_base)

linha2_reconhecida = []
for palavra in letrasLinha2_norm:
    texto = visco.reconhece_palavra(palavra, templates_norm, metrica='ZNCC')
    linha2_reconhecida.append(texto)
print(*linha2_reconhecida)

# Linha 3
I_dilatada_3 = cv2.dilate(linhas[2], kernel_linhas, iterations=1)
# plt.figure()
# plt.imshow(I_dilatada_3, cmap='gray') 
# plt.title("Dilatação Horizontal - Linha 3")
palavras_3 = visco.identificar_palavras_linha(I_dilatada_3,linhas[2])
letrasLinha3 = visco.identificar_letras(palavras_3)
#print('tam letrasLinha3',len(letrasLinha3))
letrasLinha3_norm = visco.normaliza_letras_linha(letrasLinha3, altura_base, largura_base)
#print('Tamanho letrasLinha2 normalizado:', len(letrasLinha2_norm))


linha3_reconhecida = []

for palavra in letrasLinha3_norm:
    texto = visco.reconhece_palavra(palavra, templates_norm, metrica='ZNCC')
    linha3_reconhecida.append(texto)

print(*linha3_reconhecida)


plt.show()
