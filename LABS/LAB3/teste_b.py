import cv2
import numpy as np
from matplotlib import pyplot as plt

def bounding_box(I1):
    y, x = np.where(I1)
    ymin = np.min(y)
    ymax = np.max(y)
    xmin = np.min(x)
    xmax = np.max(x)

    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])

    return p0, p1

def mpq(I1, p, q):
    y, x = np.where(I1)

    mpq = np.float32(np.sum((x**p)*(y**q)))

    return mpq

def centroide(I1):
    m00 = mpq(I1, 0, 0)
    m10 = mpq(I1, 1, 0)
    m01 = mpq(I1, 0, 1)
    
    xc = m10/m00
    yc = m01/m00

    c0 = np.array([xc, yc])

    return c0
def upq(I1, p, q):

    c0 = centroide(I1)
    xc = c0[0]
    yc = c0[1]

    y, x = np.where(I1)

    momento_centroide = np.sum(((x-xc)**p) * ((y-yc)**q))

    return momento_centroide
def analisaRegioes(I1):
    infoRegioes = []

    # rotulamento
    num_elem, I_labels = cv2.connectedComponents(I1)

    for i in np.arange(1, num_elem):
        # dados dos componentes
        dados_do_compenente = dict( )
        # imagem do componente cenectado (região de interesse)
        I_componente = np.uint8(I_labels == i)*255
        dados_do_compenente['imagem'] = I_componente

        #bounding box
        p0, p1 = bounding_box(I_componente)
        dados_do_compenente['bb_p0'] = p0
        dados_do_compenente['bb_p1'] = p1

        # area
        m00 = mpq(I_componente, 0, 0)
        dados_do_compenente['area'] = m00

        # centroide
        c0 = centroide(I_componente)
        dados_do_compenente['centroide'] = c0

        # momentos centrais de segunda ordem
        u20 = upq(I_componente,2,0)
        u02 = upq(I_componente,0,2)
        u11 = upq(I_componente,1,1)

        # matriz de inércia da região
        J = np.array([[u20, u11], [u11, u02]])

        # matriz de inércia da elipse equivalente
        Je = 4/m00 * J
        avalores, avetores = np.linalg.eig(Je)

        # raios da elipse
        raios = np.sort(np.sqrt(avalores))
        menor_raio = raios[0]
        maior_raio = raios[1]

        dados_do_compenente['maior_raio'] = maior_raio
        dados_do_compenente['menor_raio'] = menor_raio

        # orientação da elipse
        pos = np.argmax(avalores)

        vx = avetores[0, pos]
        vy = avetores[1, pos]
        orientacao = np.rad2deg(np.atan2(vy, vx))

        dados_do_compenente['razao_raios'] = menor_raio/maior_raio
        dados_do_compenente['orientacao'] = orientacao

        # extrai coordenas dos pixels de borda
        contours, hierarchy = cv2.findContours(I_componente, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

        x = contours[0][:,0,0]
        y = contours[0][:,0,1] 

        # cálculo do perímetro
        perimetro = np.sqrt((y[-1] - y[0])**2 + (x[-1]- x[0])**2)
        N = len(y)

        for n in np.arange(0,N-1):
            perimetro = perimetro + np.sqrt((y[n] - y[n+1])**2 + (x[n]- x[n+1])**2)

        dados_do_compenente['perimetro'] = perimetro

        # circularidade
        circularidade = 4*np.pi*m00/(perimetro**2)

        dados_do_compenente['circularidade'] = circularidade

        # curva de distância
        xc = c0[0]
        yc = c0[1]

        curva_distancia = np.zeros(N)
        for n in np.arange(0, N):
            curva_distancia[n] = np.sqrt((y[n]-yc)**2 + (x[n]-xc)**2)

        dados_do_compenente['curva_distancia'] = curva_distancia
        
        # adiciona o dicionário na lista infoRegioes
        infoRegioes.append(dados_do_compenente.copy())

    return infoRegioes
def desenha_boundingBoxes(I1, infoRegioes, color, thickness):
    num_regioes = len(infoRegioes)

    for i in np.arange(0, num_regioes):
        p0 = infoRegioes[i]['bb_p0']
        p1 = infoRegioes[i]['bb_p1']

        cv2.rectangle(I1, p0, p1, color, thickness)

    return I1

def desenha_centroide(I1,infoRegioes, color, radius):
    num_regioes  = len(infoRegioes)
    for i in np.arange(0, num_regioes):
        c0 = infoRegioes[i]['centroide']
        cv2.circle(I1, np.int32(c0), radius, color, -1)

    return I1


def modifica_tamanho_img(templates, altura_base, largura_base):
    templates_norm = {}
    for letra, I_temp in templates.items():
        y, x = np.where(I_temp>0)
        if len(y) == 0 or len(x) == 0:
            templates_norm[letra] = np.zeros((altura_base, largura_base), np.uint8)
            continue
        ymin, ymax = np.min(y), np.max(y)
        xmin, xmax = np.min(x), np.max(x)
        I_crop = I_temp[ymin-5:ymax+5, xmin-5:xmax+5]
        I_resize = cv2.resize(I_crop, (largura_base, altura_base), interpolation=cv2.INTER_NEAREST)
        templates_norm[letra] = I_resize

    return templates_norm

def modifica_tamanho_img_2(I_temp, altura_base, largura_base):
    y, x = np.where(I_temp > 0)
    if len(y) == 0 or len(x) == 0:
        return np.zeros((altura_base, largura_base), np.uint8)

    ymin, ymax = np.min(y), np.max(y)
    xmin, xmax = np.min(x), np.max(x)
    
    # Garante que o recorte não saia dos limites da imagem
    ymin = max(ymin - 5, 0)
    ymax = min(ymax + 5, I_temp.shape[0])
    xmin = max(xmin - 5, 0)
    xmax = min(xmax + 5, I_temp.shape[1])
    
    I_crop = I_temp[ymin:ymax, xmin:xmax]
    I_resize = cv2.resize(I_crop, (largura_base, altura_base), interpolation=cv2.INTER_NEAREST)
    
    return I_resize

def similaridade(I1, I2, metrica):
    I1 = np.float32(I1)/255
    I2 = np.float32(I2)/255

    if metrica == 'SAD':
        sim = np.sum(np.abs(I1 - I2))

    elif metrica == 'ZSAD':
        sim = np.sum(np.abs((I1 - np.mean(I1)) - (I2 - np.mean(I2))))

    elif metrica == 'SSD':
        sim = np.sum((I1 - I2) ** 2)

    elif metrica == 'ZSSD':
        sim = np.sum(((I1 - np.mean(I1)) - (I2 - np.mean(I2))) ** 2)
    elif metrica == 'NCC':
        num = np.sum(I1 * I2)
        den = np.sqrt(np.sum(I1**2) * np.sum(I2**2))
        if den != 0:
            sim = num/den
        else:
            sim = 0

    elif metrica == 'ZNCC':
        mean1 = np.mean(I1)
        mean2 = np.mean(I2)
        num = np.sum((I1 - mean1) * (I2 - mean2))
        den = np.sqrt(np.sum((I1 - mean1)**2) * np.sum((I2 - mean2)**2))
        if den != 0:
            sim = num/den
        else:
            sim = 0
    else:
        sim = np.float32('nan')

    return sim
    
def detecta_dois_pontos(infoLinha):
    dois_pontos = []
    for j, info in enumerate(infoLinha):
        area = info['area']
        if area < 150:
            print(f"Área = {area:.2f}")
            dois_pontos.append(info)
    return dois_pontos


def extrair_letras(infoLinha):
    letras_linha = []
    for info in infoLinha:
        c = info['centroide']
        x_centroid = c[0]
        letra_img = info['imagem']
        letras_linha.append((x_centroid, letra_img))
        letras_linha.sort(key=lambda tup: tup[0])# Ordena pelo eixo x (da esquerda para a direita)
    return letras_linha

def template_matching(I, template, metrica):
    M_img, N_img = I.shape
    M_template, N_template = template.shape

    I_similaridade = np.zeros((M_img-M_template+1, N_img-N_template+1), np.float32)

    for y in np.arange(0, M_img-M_template+1):
        for x in np.arange(0, N_img-N_template+1):
            janela = I[y:y+M_template, x:x+N_template]
            I_similaridade[y, x] = similaridade(janela, template, metrica)

    return I_similaridade

def reconhece_palavra(letras_palavra, templates_norm, metrica='ZNCC'):
    palavra_reconhecida = ''
    
    for x_centroid, letra_img in letras_palavra:
        melhor_letra = None
        #   melhor_sim = np.float32(10e10)
        if metrica == 'NCC' or metrica == 'ZNCC':
            melhor_sim = -np.float32(10e10)
        else:
            melhor_sim = np.float32(10e10)

        for letra, template in templates_norm.items():
                    
            I_sim = template_matching(letra_img, template, metrica)
            sim = I_sim[0,0] 
            
            if metrica in ['NCC','ZNCC']:
                if sim > melhor_sim:
                    melhor_sim = sim
                    melhor_letra = letra
            else:
                if sim < melhor_sim:
                    melhor_sim = sim
                    melhor_letra = letra
        
       
        if melhor_letra is None:
            melhor_letra = '.'
        
        palavra_reconhecida += melhor_letra
    
    return palavra_reconhecida

#--------------------------------------------------------------------------------
# ALFABETO
I_letras = cv2.imread('LABS/LAB3/banco_de_imagens/template_letras.png', cv2.IMREAD_GRAYSCALE)
_, I_bin_letras = cv2.threshold(I_letras, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


# plt.imshow(I_bin_letras, cmap='gray')
# plt.title('Imagem alfabeto binarizado')

infoLetras = analisaRegioes(I_bin_letras)

print('Área: ', infoLetras[0]['area'])
print('Coordenadas: ', infoLetras[0]['bb_p1'])
print('Coordenadas: ', infoLetras[0]['bb_p0'])

I5 = cv2.cvtColor(I_letras, cv2.COLOR_GRAY2BGR)
color = (255, 128,128)
thickness = 1
radius = 2

I5 = desenha_boundingBoxes(I5, infoLetras, color, thickness)
# plt.figure()
# plt.imshow(cv2.cvtColor(I5, cv2.COLOR_BGR2RGB))
# plt.title('Boundingbox alfabeto')

letras = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T', 'U','V','X','Z','W']
centroides = [info['centroide'][0] for info in infoLetras]
indices_ordenados = np.argsort(centroides)

templates = {}
for i in range(len(letras)):
    idx = indices_ordenados[i]
    letra = letras[i]
    templates[letra] = infoLetras[idx]['imagem']

# Normaliza para tamanho fixo (altura_base x largura_base)
#altura_base, largura_base = 46, 37
x1, y1 = infoLetras[0]['bb_p1']
x0, y0 = infoLetras[0]['bb_p0']
altura_base = y1-y0
largura_base = x1-x0

templates_norm = modifica_tamanho_img(templates, altura_base, largura_base)
'''plt.figure()
plt.imshow(templates_norm['A'], cmap='gray')  
plt.title('A')'''

'''
plt.figure(figsize=(15, 3))
i = 1
for letra, img in templates_norm.items():
    plt.subplot(3, 9, i)  # ajusta 3 linhas e 9 colunas conforme necessário
    plt.imshow(img, cmap='gray')
    plt.title(letra)
    plt.axis('off')
    i += 1
plt.tight_layout()
'''


# ----------------------------------------------------------------------------------------------------------------------
# ETIQUETAS 
I1 = cv2.imread('LABS/LAB3/banco_de_imagens/im8.png')
plt.figure()
plt.imshow(I1)
plt.title("Imagem Original")

# Segmentação ----------------------------------------------------------------------------------
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

# analisar regiões
infoRegioes = analisaRegioes(I_bin)

I_copia = I1.copy()
I_copia = desenha_boundingBoxes(I_copia, infoRegioes, (255, 0, 0), 2)
I_copia = desenha_centroide(I_copia, infoRegioes, (0, 0, 255), 5)

# plt.figure()
# plt.imshow(cv2.cvtColor(I_copia, cv2.COLOR_BGR2RGB))
# plt.title("Centroides e Bounding Boxes")

#print(infoRegioes)

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


I_vis = I1.copy()
for i, c in enumerate(centroides_organizados):
    x, y = np.int32(c)
    cv2.circle(I_vis, (x, y), 7, (0, 0, 255), -1)
    cv2.putText(I_vis, str(i), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# plt.figure()
# plt.imshow(cv2.cvtColor(I_vis, cv2.COLOR_BGR2RGB))
# plt.title("Centróides Numerados")



# HOMOGRAFIA PLANAR -----------------------------------------------------------------------------------
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
# plt.title('Imagem Alinhada')

xmargem, ymargem = 20, 20
I3 = I2[xmargem:-xmargem, ymargem:-ymargem]
# plt.figure()
# plt.imshow(I3)

I_gray = cv2.cvtColor(I3, cv2.COLOR_RGB2GRAY) 

# fazer com que a imagem seja binária
_, I_bin_txt = cv2.threshold(I_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 

# plt.figure()
# plt.imshow(I_bin_txt, cmap='gray')
# plt.title('Imagem binária')

# Identificar linhas
kernel_linha = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 3))
I_dilatada = cv2.dilate(I_bin_txt, kernel_linha, iterations=1)
# plt.figure()
# plt.imshow(I_dilatada, cmap='gray') 
# plt.title("Dilatação Horizontal (Linhas Unidas)")

num_linhas, labels_linhas = cv2.connectedComponents(I_dilatada)
print(f"Número de linhas detectadas: {num_linhas-1}")

linhas = []
for i in range(1, num_linhas):
    I_componente = np.uint8(labels_linhas == i) * 255
    y, x = np.where(I_componente)
    ymin, ymax = np.min(y), np.max(y)
    xmin, xmax = np.min(x), np.max(x)
    linha_crop = I_bin_txt[ymin:ymax+1, xmin:xmax+1]
    linhas.append(linha_crop)
    plt.figure() 
    plt.imshow(linha_crop, cmap='gray') 
    plt.title(f"Linha {i}")

print(f"Total de palavras extraídas: {len(linhas)}")

kernel_linhas = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))#30,8
I_dilatada_1 = cv2.dilate(linhas[0], kernel_linhas, iterations=1)
# plt.figure()
# plt.imshow(I_dilatada_1, cmap='gray') 
# plt.title("Dilatação Horizontal - Linha 1")

I_dilatada_2 = cv2.dilate(linhas[1], kernel_linhas, iterations=1)
# plt.figure()
# plt.imshow(I_dilatada_2, cmap='gray') 
# plt.title("Dilatação Horizontal - Linha 2")

I_dilatada_3 = cv2.dilate(linhas[2], kernel_linhas, iterations=1)
# plt.figure()
# plt.imshow(I_dilatada_3, cmap='gray') 
# plt.title("Dilatação Horizontal - Linha 3")


# INFORMAÇÕES DAS LINHAS---------------------------------------
# Linha 1 
num_linhas1, labels_linhas1 = cv2.connectedComponents(I_dilatada_1)
palavras_1 = []
for i in range(1, num_linhas1):
    I_componente = np.uint8(labels_linhas1 == i)*255
    y, x = np.where(I_componente)
    ymin, ymax = np.min(y), np.max(y)
    xmin, xmax = np.min(x), np.max(x)
    linha_crop1 = linhas[0][ymin:ymax+1, xmin:xmax+1]
    palavras_1.append(linha_crop1)
    # plt.figure() 
    # plt.imshow(linha_crop1, cmap='gray') 
    # plt.title(f"Palavra {i}, Linha 1")

#print(f"Total de palavras extraídas da linha 1: {len(palavras_1)}")

'''infoLinha1 = analisaRegioes(linhas[0])
linha_vis1 = cv2.cvtColor(linhas[0], cv2.COLOR_GRAY2BGR)

# Desenha as bounding boxes
linha_vis = desenha_boundingBoxes(linha_vis1, infoLinha1, color=(0, 255, 0), thickness=1)

# Exibe a imagem
plt.figure(figsize=(12,4))
plt.imshow(cv2.cvtColor(linha_vis, cv2.COLOR_BGR2RGB))
plt.title("Bounding Boxes das letras da Linha 1")
'''
#letras_linha1 = extrair_letras(infoLinha1)
def normaliza_letras_linha(letrasLinha1, altura_base, largura_base):
    """
    Recebe uma lista de palavras (cada palavra é lista de tuplas (x_centroid, letra_img))
    e retorna a mesma estrutura, mas com todas as imagens das letras redimensionadas.
    """
    letrasLinha1_norm = []

    for palavra in letrasLinha1:
        palavra_norm = []
        for x_centroid, letra_img in palavra:
            # encontra pixels não zero
            y, x = np.where(letra_img > 0)
            if len(y) == 0 or len(x) == 0:
                letra_redim = np.zeros((altura_base, largura_base), np.uint8)
            else:
                ymin, ymax = np.min(y), np.max(y)
                xmin, xmax = np.min(x), np.max(x)
    
                ymin = max(ymin - 5, 0)
                ymax = min(ymax + 5, letra_img.shape[0])
                xmin = max(xmin - 5, 0)
                xmax = min(xmax + 5, letra_img.shape[1])

                letra_crop = letra_img[ymin:ymax, xmin:xmax]
                letra_redim = cv2.resize(letra_crop, (largura_base, altura_base), interpolation=cv2.INTER_NEAREST)

            palavra_norm.append((x_centroid, letra_redim))

        letrasLinha1_norm.append(palavra_norm)

    return letrasLinha1_norm

letrasLinha1 = []
for i in range(len(palavras_1)):
    infoPalavra = analisaRegioes(palavras_1[i])
    infoPalavra = [info for info in infoPalavra if info['area'] >= 150] # remove dois pontos
    letras_palavra = extrair_letras(infoPalavra)
    letrasLinha1.append(letras_palavra)

#print('tam letrasLinha1',len(letrasLinha1))
letrasLinha1_norm = normaliza_letras_linha(letrasLinha1, altura_base, largura_base)
#print('Tamanho letrasLinha1 normalizado:', len(letrasLinha1_norm))

#letras_linha1norm = modifica_tamanho_img(letras_dict, altura_base, largura_base)
'''plt.figure(figsize=(12,4))
i = 1
for palavra in letrasLinha1_norm:
    for x_centroid, letra_img in palavra:
        plt.subplot(2, 10, i)  
        plt.imshow(letra_img, cmap='gray')
        plt.axis('off')
        i += 1
plt.tight_layout()
'''

linha1_reconhecida = []

for palavra in letrasLinha1_norm:
    texto = reconhece_palavra(palavra, templates_norm, metrica='ZNCC')
    linha1_reconhecida.append(texto)

#print("Linha 1 reconhecida:", ' '.join(linha1_reconhecida))
print('Linha 1 reconhecida:', *linha1_reconhecida)

#--------------------------------------------------

# Linha 2
num_linhas2, labels_linhas2 = cv2.connectedComponents(I_dilatada_2)
palavras_2 = []
for i in range(1, num_linhas2):
    I_componente = np.uint8(labels_linhas2 == i)*255
    y, x = np.where(I_componente)
    ymin, ymax = np.min(y), np.max(y)
    xmin, xmax = np.min(x), np.max(x)
    linha_crop2 = linhas[1][ymin:ymax+1, xmin:xmax+1]
    palavras_2.append(linha_crop2)
    # plt.figure() 
    # plt.imshow(linha_crop2, cmap='gray') 
    # plt.title(f"Palavra {i}, Linha 2")

#print(f"Total de palavras extraídas da linha 2: {len(palavras_2)}")


infoLinha2 = analisaRegioes(linhas[1])
linha_vis2 = cv2.cvtColor(linhas[1], cv2.COLOR_GRAY2BGR)

# Desenha as bounding boxes
linha_vis2 = desenha_boundingBoxes(linha_vis2, infoLinha2, color=(0, 255, 0), thickness=1)

# Exibe a imagem
# plt.figure(figsize=(12,4))
# plt.imshow(cv2.cvtColor(linha_vis2, cv2.COLOR_BGR2RGB))
# plt.title("Bounding Boxes das letras da Linha 2")

#letras_linha2 = extrair_letras(infoLinha2)
letrasLinha2 = []
for i in range(len(palavras_1)):
    infoPalavra = analisaRegioes(palavras_2[i])
    infoPalavra = [info for info in infoPalavra if info['area'] >= 150] # remove dois pontos
    letras_palavra = extrair_letras(infoPalavra)
    letrasLinha2.append(letras_palavra)

#print('tam letrasLinha1',len(letrasLinha2))
letrasLinha2_norm = normaliza_letras_linha(letrasLinha2, altura_base, largura_base)
#print('Tamanho letrasLinha1 normalizado:', len(letrasLinha2_norm))

#letras_linha1norm = modifica_tamanho_img(letras_dict, altura_base, largura_base)
'''plt.figure(figsize=(12,4))
i = 1
for palavra in letrasLinha2_norm:
    for x_centroid, letra_img in palavra:
        plt.subplot(2, 10, i)  
        plt.imshow(letra_img, cmap='gray')
        plt.axis('off')
        i += 1
plt.tight_layout()
'''

linha2_reconhecida = []

for palavra in letrasLinha2_norm:
    texto = reconhece_palavra(palavra, templates_norm, metrica='ZNCC')
    linha2_reconhecida.append(texto)

print('Linha 2 reconhecida:', *linha2_reconhecida)
#--------------------------------------------------
# Linha 3
num_linhas3, labels_linhas3 = cv2.connectedComponents(I_dilatada_3)
palavras_3 = []
for i in range(1, num_linhas3):
    I_componente = np.uint8(labels_linhas3 == i)*255
    y, x = np.where(I_componente)
    ymin, ymax = np.min(y), np.max(y)
    xmin, xmax = np.min(x), np.max(x)
    linha_crop3 = linhas[2][ymin:ymax+1, xmin:xmax+1]
    palavras_3.append(linha_crop3)
    # plt.figure() 
    # plt.imshow(linha_crop3, cmap='gray') 
    # plt.title(f"Palavra {i}, Linha 3")

#print(f"Total de palavras extraídas da linha 3: {len(palavras_3)}")

infoLinha3 = analisaRegioes(linhas[2])
linha_vis3 = cv2.cvtColor(linhas[2], cv2.COLOR_GRAY2BGR)
linha_vis3 = desenha_boundingBoxes(linha_vis3, infoLinha3, color=(0, 255, 0), thickness=1)

# Exibe a imagem
# plt.figure(figsize=(12,4))
# plt.imshow(cv2.cvtColor(linha_vis3, cv2.COLOR_BGR2RGB))
# plt.title("Bounding Boxes das letras da Linha 3")

#letras_linha3 = extrair_letras(infoLinha3)

letrasLinha3 = []
for i in range(len(palavras_3)):
    infoPalavra = analisaRegioes(palavras_3[i])
    infoPalavra = [info for info in infoPalavra if info['area'] >= 150] # remove dois pontos
    letras_palavra = extrair_letras(infoPalavra)
    letrasLinha3.append(letras_palavra)

#print('tam letrasLinha3',len(letrasLinha3))
letrasLinha3_norm = normaliza_letras_linha(letrasLinha3, altura_base, largura_base)
#print('Tamanho letrasLinha1 normalizado:', len(letrasLinha2_norm))

#letras_linha1norm = modifica_tamanho_img(letras_dict, altura_base, largura_base)
'''plt.figure(figsize=(12,4))
i = 1
for palavra in letrasLinha3_norm:
    for x_centroid, letra_img in palavra:
        plt.subplot(2, 10, i)  
        plt.imshow(letra_img, cmap='gray')
        plt.axis('off')
        i += 1
plt.tight_layout()'''


linha3_reconhecida = []

for palavra in letrasLinha3_norm:
    texto = reconhece_palavra(palavra, templates_norm, metrica='ZNCC')
    linha3_reconhecida.append(texto)

print('Linha 3 reconhecida:', *linha3_reconhecida)

#----------------------------------------------------------------------------------
'''lin_I1,  col_I1 = .shape
lin_template,  col_template = T1.shape

n_linhas = lin_I1 - lin_template +1
n_colunas = col_I1 - col_template +1

I_similaridade = np.zeros((n_linhas, n_colunas), np.float32)

for x in np.arange(0, col_I1 - col_template + 1):
    for y in np.arange(0, lin_I1-lin_template+1):

        janela = I1[y:y+lin_template, x:x+col_template] 

        I_similaridade[y,x] = similaridade(janela, T1)
'''

plt.show()