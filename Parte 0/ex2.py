import numpy as np

a = np.array([2.5, 6.5, 87, 4.5, 9])
print(a)

# Indexação
print(a[0]) # acessa o primeiro valor
print(a[:3]) # do primeiro ao terceiro
print(a[-1]) # acessa o último valor
print(a[:-1]) #acessa todos os valores até o penúltimo

print(f'{a.size} elementos no array')
print(f'Foramto do array {a.shape}')
print(f'Formato dos elementos do array: {a.dtype}')


# Array de 2 dimensões
B = np.array([[1, 2, 3],[4, 5, 6]])
print(B)
# indexação
print(B[0,1]) #indice da 1° dimensão, indice da 2° dimensão
print(B[0,:])
print(B[:,1])

print(B[:,0:2])

# funções para a criação de arrays
c = np.zeros(3) #array de 1d com elementos zeros
print(c)
print(c.dtype)

d = np.zeros((2,3))
print(d)
print(d.dtype)

e = np.ones(5)
print(e)

f = np.arange(0,9)
print(f)

g = np.arange(10)
print(g)

# gerar arrays aleatórios
h = np.random.randn(2,3)
print(h)

# operações aritméticas
i = np.array([[1,2],[3,4]])
j = np.array([[6,7],[8,9]])

print('Soma: ', i+j)
print('Multiplicação ponto a ponto:', i*j)
print('Multiplicação matricial', np.dot(i,j))

# Solução de sistemas de equações lineares
A = np.array([[1,2],[3,4]])
b = np.array([9,8])

x = np.linalg.solve(A,b)
print(f'Solução da eq: {x}')

