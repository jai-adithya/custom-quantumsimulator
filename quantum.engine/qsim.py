import numpy as np
from numpy import sqrt, exp, pi, cos, sin

#Defining all the quantum gates 

# H creates superposition / swaps between + and 0 states or 1 and - states
H = np.array([[1,  1],
              [1, -1]], dtype=complex) / np.sqrt(2)

# X gate - NOT gate - Bit flip
X = np.array([[0, 1],
              [1, 0]], dtype=complex)    

# Y gate - X and Y together - Bit and phase flip
Y = np.array([[0, -1j],
              [1j, 0]], dtype=complex)

# Z gate - Phase flip
Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

# 90° phase shift
S = np.array([[1, 0],
              [0, -1j]], dtype=complex)

# 45° phase shift
T = np.array([[1, 0], [0, exp(1j * pi / 4)]], dtype=complex)


# below are the XYZ gates with variable amounts of rotation theta
def RX(theta) :
    c, s = cos(theta/2), sin(theta/2)
    return np.array([[c, -s*1j], 
                     [-1j*s, c]], dtype=complex)

def RY(theta) :
    c, s = cos(theta/2), sin(theta/2)
    return np.array([[c, -s], 
                     [s, c]], dtype=complex)

def RZ(theta) :
    return np.array([[exp(-1j * theta/2), 0],
                     [0, exp(1j * theta/2)]], dtype=complex)




