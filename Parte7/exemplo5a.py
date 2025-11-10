import cv2
import numpy as np
from matplotlib import pyplot as plt

# Funções -----------------------------------------------------------
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

def upq(I1, p, q):

    c0 = centroide(I1)
    xc = c0[0]
    yc = c0[1]

    y, x = np.where(I1)

    momento_centroide = np.sum(((x-xc)**p) * ((y-yc)**q))

    return momento_centroide


# --------------------------------------------------------------------

I1 = cv2.imread('./Imagens/sharks.png',cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(I1, cmap='gray')
plt.title('Imagem Original')
# extrai características de região
infoRegioes = analisaRegioes(I1)

indice1 = 1
indice2 = 0

plt.figure()
plt.imshow(infoRegioes[indice1]['imagem'], cmap='gray')


plt.figure()
plt.imshow(infoRegioes[indice2]['imagem'], cmap='gray')


curva1 = infoRegioes[indice1]['curva_distancia']
curva2 = infoRegioes[indice2]['curva_distancia']

plt.figure()
plt.plot(curva1)
plt.plot(curva2)

# Interpolação
N = 400
N1 = len(curva1)
N2 = len(curva2)

curva1i = np.interp(np.linspace(0, N1-1, N), np.arange(0, N1), curva1)
curva2i = np.interp(np.linspace(0, N2-1, N), np.arange(0, N2), curva2)

plt.figure()
plt.plot(curva1i)
plt.plot(curva2i)

# remoção da média
y1 = curva1i - np.mean(curva1i)
y2 = curva2i - np.mean(curva2i)

plt.figure()
plt.plot(y1)
plt.plot(y2)

# normalização da amplitude
y1n = y1/ np.sqrt(np.sum(y1*y1))
y2n = y2/ np.sqrt(np.sum(y2*y2))

plt.figure()
plt.plot(y1n)
plt.plot(y2n)

# Correlação
correlacao = np.zeros(N)

for k in np.arange(0, N):
   correlacao[k] = np.sum(np.roll(y1n, k)*y2n)
plt.figure()
plt.plot(correlacao)
plt.grid()
plt.title('Correlação')

similaridade = np.max(correlacao)
print(f'Similaridade entre as regiões = {similaridade}')

plt.show()