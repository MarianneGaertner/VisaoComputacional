import numpy as np

arr = np.arange(12)
print("Array original:", arr)

arr_reshape = arr.reshape((3, 4))
print("Array reshaped para 3x4:\n", arr_reshape)

arr_ravel = arr_reshape.ravel()
print("Array flatten:\n", arr_ravel)