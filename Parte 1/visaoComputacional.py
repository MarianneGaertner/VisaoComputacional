import cv2
import numpy as np

def limiarizacao_global1(I,L):
    n_linhas, n_colunas = I.shape
    O = np.zeros((n_linhas, n_colunas), np.uint8)

    for x in np.arange(0, n_colunas):
        for y in np.arange(0, n_linhas):
            if I[y,x] >= L:
                O[y,x] = 255

    return O

def limiarizacao_global2(I,L):
    n_linhas, n_colunas = I.shape
    O = np.zeros((n_linhas, n_colunas), np.uint8)

    indices = I >= L
    O[indices] = 255
    return O

