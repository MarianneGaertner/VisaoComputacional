import cv2
import numpy as np

# Limiarização global (conversao de imagem em escala de cinza para 
# binaria). Algoritmo: compara cada pixel da imagem em escala de 
# cinza com um limiar (definido pelo projetista). Se o valor do 
# pixel na imagem em escala de cinza for maior do o limiar, na 
# imagem binaria de saida eh atribuido a cor branca (255) no 
# pixel correspondente. Caso contrario eh atruido preto (0).
def limiarizacao_global_1(I, limiar):
    n_linhas, n_colunas = I.shape

    I_bin = np.zeros((n_linhas, n_colunas), np.uint8)

    for y in np.arange(0, n_linhas):
        for x in np.arange(0, n_colunas):
            if I[y,x] > limiar:
                I_bin[y,x] = 255
    
    return I_bin

def limiarizacao_global_2(I, limiar):
    n_linhas, n_colunas = I.shape

    I_bin = np.zeros((n_linhas, n_colunas), np.uint8)

    # Acessa os elementos de I_bin por indexação lógica
    indices = (I > limiar) 
    I_bin[indices] = 255

    return I_bin

# calcula o histograma da imagem I em escala de cinza
def imhist(I):

    hist = np.zeros(256)

    n_linhas, n_colunas = I.shape

    for y in np.arange(0, n_linhas):
        for x in np.arange(0, n_colunas):
            pixel_value = I[y,x]
            hist[pixel_value] = hist[pixel_value] + 1

    return hist


def color2bin_1(I_bgr, cor_referencia, delta):

    ref_azul = cor_referencia[0]
    ref_verde = cor_referencia[1]
    ref_vermelho = cor_referencia[2]

    B = I_bgr[:,:,0]
    G = I_bgr[:,:,1]
    R = I_bgr[:,:,2]

    n_linhas, n_colunas, n_camanas = I_bgr.shape

    Mr = np.zeros((n_linhas, n_colunas), np.uint8)
    Mg = np.zeros((n_linhas, n_colunas), np.uint8)
    Mb = np.zeros((n_linhas, n_colunas), np.uint8)

    for y in np.arange(0, n_linhas):
        for x in np.arange(0, n_colunas):
            
            if (R[y,x] >= ref_vermelho - delta) &\
                (R[y,x] <= ref_vermelho + delta):
                Mr[y,x] = 255

            if (G[y,x] >= ref_verde - delta) &\
                (G[y,x] <= ref_verde + delta):
                Mg[y,x] = 255

            if (B[y,x] >= ref_azul - delta) &\
                (B[y,x] <= ref_azul + delta):
                Mb[y,x] = 255

    I_aux = cv2.bitwise_and(Mr, Mg)
    I_bin = cv2.bitwise_and(I_aux, Mb)

    return I_bin

def color2bin_2(I_bgr, cor_referencia, delta):

    ref_azul = cor_referencia[0]
    ref_verde = cor_referencia[1]
    ref_vermelho = cor_referencia[2]

    B = I_bgr[:,:,0]
    G = I_bgr[:,:,1]
    R = I_bgr[:,:,2]

    B_indices_logicos = (B >= (ref_azul-delta)) &\
          (B <= (ref_azul+delta))
    G_indices_logicos = (G >= (ref_verde-delta)) &\
          (G <= (ref_verde+delta))
    R_indices_logicos = (R >= (ref_vermelho-delta)) &\
          (R <= (ref_vermelho+delta))

    Indices_logicos = B_indices_logicos & G_indices_logicos & R_indices_logicos

    n_linhas, n_colunas, n_camanas = I_bgr.shape
    I_bin = np.zeros((n_linhas, n_colunas), np.uint8)
    I_bin[Indices_logicos] = 255

    return I_bin

def imresize(I, M, N):

    # imagem de saída
    O = np.zeros((M,N), np.uint8)

    # matriz de mapeamento inverso
    n_linhas, n_colunas = I.shape
    Ai = np.array([[n_colunas/N, 0], [0, n_linhas/M]])

    # realiza o mapeamento inverso
    for yl in np.arange(0, M):
        for xl in np.arange(0, N):
            pixel_position = np.array([xl, yl])
            x, y = np.dot(Ai, pixel_position)
            x1 = int(np.floor(x))
            x2 = int(np.ceil(x))
            y1 = int(np.floor(y))
            y2 = int(np.ceil(y))

            # verifica se os valores das coordenadas anteriores são válidas
            x1, x2 = ajusta_coordenadas_pixels(x1, x2, 0, n_colunas-1)
            y1, y2 = ajusta_coordenadas_pixels(y1, y2, 0, n_linhas-1)

            A = I[y1,x1]
            B = I[y1,x2]
            C = I[y2,x1]
            D = I[y2,x2]
            
            E = ((x2 - x) * A + (x - x1) * B)/(x2 - x1)
            F = ((x2 - x) * C + (x - x1) * D)/(x2 - x1)
            G = ((y2 - y) * E + (y - y1) * F)/(y2 - y1)

            O[yl,xl] = np.uint8(G)

    return O

def ajusta_coordenadas_pixels(x1, x2, x_min, x_max):

    if x1 < x_min:
        x1 = x_min

    if x2 > x_max:
        x2 = x_max
    
    if x1 == x2:
        if x2 >= x_max:
            x2 = x_max
            x1 = x_max-1
        elif x1 <= x_min:
            x1 = x_min
            x2 = x_min+1
        else:
            x2 = x1 + 1

    return x1, x2


# Buffer simplificado para armazenar frames de vídeo
class videoBuffer:

    def __init__(self, image_shape, tamanho):
        self.tamanho = tamanho
        self.inicio = self.tamanho-1
        self.final =  0
        self.buffer = np.zeros((image_shape[0], image_shape[1], tamanho))

    def insereFrame(self, frame):
        self.inicio += 1
        if self.inicio == self.tamanho:
            self.inicio = 0
        
        self.final += 1
        if self.final == self.tamanho:
            self.final = 0

        self.buffer[:,:,self.inicio] = frame

    def primeiroFrame(self):
        return self.buffer[:,:,self.inicio]
    
    def ultimoFrame(self):
        return self.buffer[:,:,self.final]