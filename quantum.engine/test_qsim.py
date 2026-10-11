import numpy as np
from numpy import pi
from qsim import H, X, Y, Z, S, T, RX, RY, RZ, zero_state, apply_gate, CNOT, probabilities, measure, bloch_vector, simulate

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

# probability of bell state HAS to be 1/2 for 00 and 1/2 for 11
def test_probabilities_bell():
    state = CNOT(apply_gate(zero_state(2), H, 0, 2), 0, 1, 2)
    assert np.allclose(probabilities(state), [0.5, 0, 0, 0.5])

# probabilities of any state must always sum to 1
def test_probabilities_sum_to_one():
    state = apply_gate(zero_state(3), H, 1, 3)
    assert np.isclose(np.sum(probabilities(state)), 1)

# measuring bell state should always only give 00 and 11 as outputs, this test makes sure 01 and 10 dont get measured and that the number of each is roughly equal
def test_measure_bell_only_00_and_11():
    state = CNOT(apply_gate(zero_state(2), H, 0, 2), 0, 1, 2)
    counts = measure(state, 1000, 2, seed=42)
    assert set(counts) <= {"00", "11"}       # no "01" or "10" ever
    assert 400 < counts["00"] < 600          # roughly half

# doing the X gate on 00 should ONLY give 01 after measurement.
def test_measure_certain_outcome():
    state = apply_gate(zero_state(2), X, 0, 2)    # |01⟩
    counts = measure(state, 100, 2, seed=1)
    assert counts == {"01": 100}


# all the tests to see if the bloch line function I defined is working
def test_bloch_zero():
    # |0⟩ points straight up
    assert np.allclose(bloch_vector(zero_state(1), 0, 1), [0, 0, 1])

def test_bloch_plus():
    # H|0⟩ = |+⟩ points along x
    state = apply_gate(zero_state(1), H, 0, 1)
    assert np.allclose(bloch_vector(state, 0, 1), [1, 0, 0])

def test_bloch_plus_i():
    # S·H|0⟩ = |+i⟩ points along y
    state = apply_gate(apply_gate(zero_state(1), H, 0, 1), S, 0, 1)
    assert np.allclose(bloch_vector(state, 0, 1), [0, 1, 0])

def test_bloch_bell_is_centre():
    # an entangled qubit sits at the centre
    state = CNOT(apply_gate(zero_state(2), H, 0, 2), 0, 1, 2)
    assert np.allclose(bloch_vector(state, 0, 2), [0, 0, 0])

def test_bloch_qubit_order():
    # |10⟩: qubit 1 points down, qubit 0 points up
    state = apply_gate(zero_state(2), X, 1, 2)
    assert np.allclose(bloch_vector(state, 1, 2), [0, 0, -1])
    assert np.allclose(bloch_vector(state, 0, 2), [0, 0, 1])

def test_entanglement_shrinks_arrow():
    # RY(θ) then CNOT: qubit 0's arrow is (0, 0, cos θ)
    for theta in [0, np.pi / 4, np.pi / 2]:
        state = CNOT(apply_gate(zero_state(2), RY(theta), 0, 2), 0, 1, 2)
        assert np.allclose(bloch_vector(state, 0, 2), [0, 0, np.cos(theta)])

state = apply_gate(apply_gate(zero_state(1), H, 0, 1), S, 0, 1)
print(bloch_vector(state, 0, 1))