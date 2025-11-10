import cv2
import numpy as np
from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------

def modifica_tamanho_img(templates, altura_base, largura_base):
    templates_norm = {}
    for letra, I_temp in templates.items():
        
        y, x = np.where(I_temp>0)
        
        ymin, ymax = np.min(y), np.max(y)
        xmin, xmax = np.min(x), np.max(x)
        
        I_cort = I_temp[ymin-5:ymax+5, xmin-5:xmax+5]
        I_resize = cv2.resize(I_cort, (largura_base, altura_base))
        templates_norm[letra] = I_resize

    return templates_norm

def normaliza_letras_linha(letrasLinha1, altura_base, largura_base):
  
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
    
                ymin = max(ymin-5, 0)
                ymax = min(ymax+5, letra_img.shape[0])
                xmin = max(xmin-5, 0)
                xmax = min(xmax+5, letra_img.shape[1])

                letra_crop = letra_img[ymin:ymax, xmin:xmax]
                letra_redim = cv2.resize(letra_crop, (largura_base, altura_base))

            palavra_norm.append((x_centroid, letra_redim))

        letrasLinha1_norm.append(palavra_norm)

    return letrasLinha1_norm

def extrair_letras(infoLinha):
    letras_linha = []
    for info in infoLinha:
        c = info['centroide']
        x_centroid = c[0]
        letra_img = info['imagem']
        letras_linha.append((x_centroid, letra_img))
        letras_linha.sort(key=lambda tup: tup[0])# Ordena pelo eixo x (da esquerda para a direita)
    return letras_linha


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

def template_matching(I, template, metrica):
    M_img, N_img = I.shape
    M_template, N_template = template.shape

    I_similaridade = np.zeros((M_img-M_template+1, N_img-N_template+1), np.float32)

    for y in np.arange(0, M_img-M_template+1):
        for x in np.arange(0, N_img-N_template+1):
            janela = I[y:y+M_template, x:x+N_template]
            I_similaridade[y, x] = similaridade(janela, template, metrica)
    
    return I_similaridade

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


def trata_dois_pontos(infoRegioes, area_min):
    nova_info = []
    for info in infoRegioes:
        if info['area'] >= area_min:
            nova_info.append(info)
    infoPalavra = nova_info
    return infoPalavra

def identificar_linhas(I_bin_txt, kernel_linha=(70,3)):
    kernel_linha = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 3))
    I_dilatada = cv2.dilate(I_bin_txt, kernel_linha, iterations=1)
    # plt.figure()
    # plt.imshow(I_dilatada, cmap='gray') 
    # plt.title("Dilatação Horizontal")

    num_linhas, labels_linhas = cv2.connectedComponents(I_dilatada)
    #print(f"Número de linhas detectadas: {num_linhas-1}")

    linhas = []
    for i in range(1, num_linhas):
        I_componente = np.uint8(labels_linhas == i) * 255
        y, x = np.where(I_componente)
        ymin, ymax = np.min(y), np.max(y)
        xmin, xmax = np.min(x), np.max(x)
        linha_cort = I_bin_txt[ymin:ymax+1, xmin:xmax+1]
        linhas.append(linha_cort)
        # plt.figure() 
        # plt.imshow(linha_cort, cmap='gray') 
        # plt.title(f"Linha {i}")
    
    return linhas

def identificar_palavras_linha(I_dilatada,linha):
    num_linhas, labels_linhas = cv2.connectedComponents(I_dilatada)

    palavras = []
    for i in range(1, num_linhas):
        I_componente = np.uint8(labels_linhas == i)*255
        y, x = np.where(I_componente)
        ymin, ymax = np.min(y), np.max(y)
        xmin, xmax = np.min(x), np.max(x)
        linha_crop = linha[ymin:ymax+1, xmin:xmax+1]
        palavras.append(linha_crop)
        # plt.figure() 
        # plt.imshow(linha_crop, cmap='gray') 
        # plt.title(f"Palavra {i}")

    return palavras

def identificar_letras( palavras_linha):
    letrasLinha = []
    for i in range(len(palavras_linha)):
        infoPalavra = analisaRegioes(palavras_linha[i])
        infoPalavra = trata_dois_pontos(infoPalavra, area_min=150)
        letras_palavra = extrair_letras(infoPalavra)
        letrasLinha.append(letras_palavra)

    return letrasLinha



