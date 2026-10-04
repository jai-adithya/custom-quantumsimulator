import numpy as np
from numpy import pi
from qsim import H, X, Y, Z, S, T, RX, RY, RZ

def is_unitary(U):
    return np.allclose(U.conj().T @ U, np.eye(2))

def test_fixed_gates_are_unitary():
    for gate in [X, Y, Z, S, T, H]:
        assert is_unitary(gate)

def test_rotation_gates_are_unitary():
    for angles in [0, pi/2, pi/4, pi/8, 1.6742] :
        for gates in [RX, RY, RZ]:
            assert is_unitary(gates(angles))

def test_HH_is_identity():
    assert np.allclose(H @ H, np.eye(2))

def test_hzh_is_x():
    assert np.allclose(H @ Z @ H, X)

def test_x_flips_bit():
    assert np.allclose(X @ [1, 0], [0, 1])

def test_rotationX_work_properly():
        assert np.allclose(RX(pi), -1j * X)

def test_rotationY_work_properly():
        assert np.allclose(RY(pi), -1j * Y)

def test_rotationZ_work_properly():
        assert np.allclose(RZ(pi), -1j * Z)

print(H @ H)

for name, gate in [("X", X), ("Y", Y), ("Z", Z), ("H", H), ("S", S), ("T", T)]:
    print(name, gate.shape)