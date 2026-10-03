import numpy as np
from numpy import sqrt, exp, pi
#Defining all the quantum gates 

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

Hadarmard, H = np.array([[1, 1],
                         [1, -1]], dtype=complex) / sqrt(2)

Y = np.array([[0, -1j],
              [1j, 0]], dtype=complex)

Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

S = np.array([[1, 0],
              [0, -1j]], dtype=complex)

T = np.array([[1, 0],
              [0, exp(1j * pi / 4)]])