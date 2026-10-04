import numpy as np
from numpy import pi
from qsim import H, X, Y, Z, S, T, RX, RY, RZ, zero_state, apply_gate, CNOT

# testing if all the main gates are unitary matrices
def is_unitary(U):
    return np.allclose(U.conj().T @ U, np.eye(2))

def test_fixed_gates_are_unitary():
    for gate in [X, Y, Z, S, T, H]:
        assert is_unitary(gate)

def test_rotation_gates_are_unitary():
    for angles in [0, pi/2, pi/4, pi/8, 1.6742] :
        for gates in [RX, RY, RZ]:
            assert is_unitary(gates(angles))

# tests if Hadarmard applied twice will result in the identity matrice
def test_HH_is_identity():
    assert np.allclose(H @ H, np.eye(2))

def test_hzh_is_x():
    assert np.allclose(H @ Z @ H, X)

def test_x_flips_bit():
    assert np.allclose(X @ [1, 0], [0, 1])

# tests for variable rotation gates
def test_rotationX_work_properly():
        assert np.allclose(RX(pi), -1j * X)

def test_rotationY_work_properly():
        assert np.allclose(RY(pi), -1j * Y)

def test_rotationZ_work_properly():
        assert np.allclose(RZ(pi), -1j * Z)

# tests to see if the apply gate function is working perfectly
def test_x_on_qubit_0():
    state = apply_gate(zero_state(2), X, 0, 2)
    assert np.allclose(state, [0, 1, 0, 0])      # |01⟩

def test_x_on_qubit_1():
    state = apply_gate(zero_state(2), X, 1, 2)
    assert np.allclose(state, [0, 0, 1, 0])      # |10⟩

def test_h_on_qubit_0():
    state = apply_gate(zero_state(2), H, 0, 2)
    r = 1 / np.sqrt(2)
    assert np.allclose(state, [r, r, 0, 0])      # (|00⟩ + |01⟩)/√2

def test_three_qubits():
    state = apply_gate(zero_state(3), X, 2, 3)
    assert np.allclose(state, [0, 0, 0, 0, 1, 0, 0, 0])   # |100⟩

# testing CNOT
def test_cnot_does_nothing_when_control_is_0():
    state = CNOT(zero_state(2), 0, 1, 2)
    assert np.allclose(state, [1, 0, 0, 0])            # |00⟩ stays |00⟩

def test_cnot_flips_when_control_is_1():
    state = apply_gate(zero_state(2), X, 0, 2)         # |01⟩: qubit 0 is 1
    state = CNOT(state, 0, 1, 2)
    assert np.allclose(state, [0, 0, 0, 1])            # |11⟩

def test_cnot_reversed():
    state = apply_gate(zero_state(2), X, 1, 2)         # |10⟩: qubit 1 is 1
    state = CNOT(state, 1, 0, 2)                 # control 1, target 0
    assert np.allclose(state, [0, 0, 0, 1])            # |11⟩

def test_bell_state():
    state = apply_gate(zero_state(2), H, 0, 2)
    state = CNOT(state, 0, 1, 2)
    r = 1 / np.sqrt(2)
    assert np.allclose(state, [r, 0, 0, r])            # (|00⟩ + |11⟩)/√2

def test_ghz_state():
    state = apply_gate(zero_state(3), H, 0, 3)
    state = CNOT(state, 0, 1, 3)
    state = CNOT(state, 0, 2, 3)
    r = 1 / np.sqrt(2)
    assert np.allclose(state, [r, 0, 0, 0, 0, 0, 0, r])   # (|000⟩ + |111⟩)/√2