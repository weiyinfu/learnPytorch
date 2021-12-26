import numpy as np

a = np.arange(5)
w = np.ones(8)
for mode in ('full', 'same', 'valid'):
    b = np.convolve(a, w, mode)
    print(mode, b)
