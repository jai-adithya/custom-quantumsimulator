import numpy as np
from numpy import sqrt, exp, pi, cos, sin
from functools import reduce
from collections import Counter

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



I = np.eye(2, dtype=complex) #identity matrice

# function to create a zero quantum state vector of variable length
def zero_state(n):
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1
    return state

# function to apply a gate onto a quantum state vector
def apply_gate(state, gate, target, no_of_qubits):
    identities = [I] * no_of_qubits
    identities[no_of_qubits - 1 - target] = gate
    bigmatrice = reduce(np.kron, identities)
    return bigmatrice @ state

# Projectors used for CNOT gates
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)

# function for CNOT gate (used for entanglement, v.i) 
def CNOT(state, control, target, no_of_qubits) :
    # if control = 0, do nothing
    term0 = [I] * no_of_qubits
    term0[no_of_qubits - 1 - control] = P0

    # if control = 1, apply X onto the target
    term1 = [I] * no_of_qubits
    term1[no_of_qubits - 1 - control] = P1
    term1[no_of_qubits - 1 - target] = X

    cnot = reduce(np.kron, term0) + reduce(np.kron, term1)
    return cnot @ state

# function for Born Rule (Probability of a state is square of the amplitudes)
def probabilities(state):
    return (np.abs(state)**2)

# probabilty of bell state should be 0.5 for both 0s and both 1s, Bell state built using my own CNOT function
bell_state = CNOT(apply_gate(zero_state(2), H, 0, 2), 0, 1, 2)
print(probabilities(bell_state))

# this is a function to take random values and measure
def measure(state, shots, n, seed=None):
    rng = np.random.default_rng(seed)
    probs = probabilities(state)
    print(probs)
    indices = rng.choice(len(state), size=shots, p=probs)
    bitstrings = [format(i, f"0{n}b") for i in indices]
    return dict(Counter(bitstrings))

print (measure(bell_state, 1000, 2))

   




