import cv2
import numpy as np
from matplotlib import pyplot as plt


def canny(I,sigma,t_inferior,t_superior):
    lin, col = I.shape

    # PASSO 1
    #sigma=(w-1)/6
    w = int(6*sigma+1)
    kernel = np.zeros((w,w), np.float32)

    frac_k = 1/(2*np.pi*(sigma**2))
    frac_exp = - 1/(2*(sigma**2))

    for y in np.arange(0, w):
        for x in np.arange(0, w):
            j = y - (w-1)/2
            i = x - (w-1)/2
            kernel[y,x] = frac_k*np.exp(frac_exp*(j**2 + i**2))

            kernel = kernel/np.sum(kernel)
    Is = cv2.filter2D(I, -1, kernel, borderType=cv2.BORDER_REFLECT101)
    plt.figure()
    plt.imshow(Is,cmap='gray')
    plt.title("Is")

    # PASSO 2 - computar a magnitude M e a fase alfa do gradiente de Is
    #dy, dx = np.gradient(Is)
    Kx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]], np.float32)
    dx = cv2.filter2D(Is, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)
    
    Ky = np.transpose(Kx)
    dy = cv2.filter2D(Is, cv2.CV_32F, Ky, borderType=cv2.BORDER_REFLECT101)
    
    mag = np.sqrt(dx**2 + dy**2)
    #alfa = np.arctan2(dy,dx)*180/np.pi
    alfa = np.rad2deg(np.arctan2(dy,dx))
   
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mag, cmap='gray')
    plt.title("Passo 2 - Magnitude (M)")
    plt.subplot(1, 2, 2)
    #plt.imshow(alfa, cmap='jet')
    plt.imshow(alfa, cmap='hsv')
    plt.title("Passo 2 - Direção (α em graus)")

    # PASSO 3 - Supressão de não máximos locais

    Gn = np.zeros([lin, col], np.float32)
    ang = alfa.copy()
    
    ang[ang<0] += 180

    for v in range(1, lin-1):
        for u in range(1, col-1):
            
            # 0°
            if(0 <= ang[v,u]< 22.5) or (157.5<= ang[v,u] <= 180):
                ant = mag[v, u-1]
                pos = mag[v, u+1]
            # 45°
            elif(22.5 <= ang[v,u] < 67.5):
                ant = mag[v-1, u+1]
                pos = mag[v+1, u-1]
            # 90°
            elif(67.5 <= ang[v,u] < 112.5):
                ant = mag[v-1, u]
                pos = mag[v+1, u]
            # 135°
            elif(112.5 <= ang[v,u]<157.5):
                ant = mag[v-1, u-1]
                pos = mag[v+1, u+1]

            if((mag[v,u]>=pos) and (mag[v,u] >= ant)):
                Gn[v,u] = mag[v,u]
            else:
                Gn[v,u] = 0

    plt.figure()
    plt.imshow(Gn,cmap='gray')
    plt.title("Gn")
    # PASSO 4 - Processo de limiarização duplo
    GNH = (Gn >= t_superior)*1
    GNL = (Gn >= t_inferior)*1
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(GNH, cmap='gray')
    plt.title("Passo 4 - GNH")
    plt.subplot(1, 2, 2)
    plt.imshow(GNL, cmap='gray')
    plt.title("Passo 4 - GNL")
    
    GNL = GNL-GNH
    
    plt.figure()
    plt.imshow(GNL,cmap='gray')
    plt.title("GNL2")
    # PASSO 5 - Análise de conectividade
    Ib = np.copy(GNH)
    '''
    for v in range(1, lin-1):
        for u in range(1, col-1):
            if(GNH[v,u]>0):
                for i in range(-1,2):
                    for j in range(-1,2):
                        if GNL[v+i, u+j] > 0:
                            Ib[v+i, u+j] = 1
    #Ib[Ib>0] = 1
    '''
    for v in range(1, lin-1):
        for u in range(1, col-1):
            if(GNH[v,u]>0):
                for i in range(-1,2):
                    for j in range(-1,2):
                        if GNL[v+i, u+j] > 0:
                            Ib[v+i, u+j] = 1 

    #GNL[Ib == 0] = 0    
    Ib = GNH + GNL
    #Ib[Ib > 0] = 1
     
    
    plt.figure()
    plt.imshow(GNL,cmap='gray')
    plt.title("GNLpasso5")
    return Ib


I=cv2.imread("LABS/LAB2/castle.jpg",cv2.IMREAD_GRAYSCALE)
w = 3
sigma = (w-1)/6
ti = 55#55
ts = 110#110

IB = canny(I, sigma, ti,ts)
'''plt.figure()
plt.imshow(I, cmap='gray')
'''

plt.figure()
plt.imshow(IB, cmap='gray')
plt.title("Ib")
'''
plt.figure()
plt.imshow(cv2.Canny(I,10,50), cmap='gray')
plt.title("Função OpenCv")
'''
plt.show()