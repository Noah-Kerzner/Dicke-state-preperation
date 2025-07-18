# Read paper at https://arxiv.org/abs/2507.13308

import cirq
import numpy as np
from scipy.special import comb


# ------------------------------
# Generalized multi-level gates (dit gates)
# ------------------------------

class ditplus(cirq.Gate):
    """Shifts ancilary dit up of diminsion k by amount alpha"""
    def __init__(self, k, alpha):
        super().__init__()
        self.k = k
        self.alpha = alpha

    def _qid_shape_(self):
        return (self.k + 1,)

    def _unitary_(self):
        dim = self.k + 1
        mat = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim - self.alpha):
            mat[i + self.alpha, i] = 1
        mat[0, dim - 1] = 1
        return mat

    def _circuit_diagram_info_(self, args):
        return "ditplus"

class ditminus(cirq.Gate):
    """Shifts ancilary dit down of diminsion k by amount alpha"""
    def __init__(self, k, alpha):
        super().__init__()
        self.k = k
        self.alpha = alpha

    def _qid_shape_(self):
        return (self.k + 1,)

    def _unitary_(self):
        dim = self.k + 1
        mat = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(self.alpha, dim):
            mat[i - self.alpha, i] = 1
        mat[dim - 1, 0] = 1
        return mat

    def _circuit_diagram_info_(self, args):
        return "ditminus"


# ------------------------------
# Dit Rotation gate between levels i and j
# ------------------------------

class ditrotation(cirq.Gate):
    def __init__(self, theta, i, j, d):
        if i == j or not (0 <= i < d) or not (0 <= j < d):
            raise ValueError("Indices i and j must be different and in range [0, d-1]")
        super().__init__()
        self.theta = theta
        self.i = i
        self.j = j
        self.d = d

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        U = np.eye(self.d, dtype=np.complex128)
        c, s = np.cos(self.theta / 2), np.sin(self.theta / 2)
        U[self.i, self.i] = c
        U[self.j, self.j] = c
        U[self.i, self.j] = -s
        U[self.j, self.i] = s
        return U

    def _circuit_diagram_info_(self, args):
        return f"R{self.i}{self.j}({self.theta:.2f})"


# ------------------------------
# Custom two-qudit gate: CSUM1MinusX
# ------------------------------

class ControlledShiftOneMinusC(cirq.Gate):
    def __init__(self, dim_control, dim_target, power=1):
        self.d_c = dim_control
        self.d_t = dim_target
        self.power = power

    def num_qubits(self):
        return 2

    def _qid_shape_(self):
        return (self.d_c, self.d_t)

    def _unitary_(self):
        d_c, d_t = self.d_c, self.d_t
        size = d_c * d_t
        U = np.zeros((size, size), dtype=complex)

        for c in range(d_c):
            shift = (self.power * (1 - c)) % d_t
            for t in range(d_t):
                i = c * d_t + t
                j = c * d_t + (t + shift) % d_t
                U[i, j] = 1
        return U

    def _circuit_diagram_info_(self, args):
        arrow = "+" if self.power >= 0 else "-"
        return [f"ctrl(d={self.d_c})", f"{arrow}(1-c)*{abs(self.power)} mod {self.d_t}"]



# ------------------------------
# Dit Gamma Function
# ------------------------------

def ditgamma(n, k, i, l, m, s):
    """ The gamma for the generalized qudit. 

    Args:
        n (int): number of working qudits
        k (int): number of spin operators
        i (int): the curent working qudit number
        l (int): the current I gate number
        m (int): the value of the qudit
        s (float): the spin of the system

    Returns:
        float: gamma for working qudit
    """    
    if k - l <= 2 * s * (n - i + 1): 
        num = comb(2 * s * (n - i), k - l - m) * comb(2 * s, m)
        denom = comb(2 * s * (n - i + 1), k - l)
        return np.sqrt(num / denom)
    return 0.0


def list_prod(arr):
    """Helper function for angle calculations 

    Args:
        arr (float): list of already calculated thetas

    Returns:
        float: helper product
    """    
    ret = 1
    for arg in arr:
        ret *= np.sin(arg / 2)
    return ret

def get_dit_thetas(n, k, i, l, s):
    """
    Calculates all thetas for general rotations 
    Args:
        n (int): number of working qudits
        k (int): number of spin operators
        i (int): the curent working qudit number
        l (int): the current I gate number
        s (float): the spin of the system


    Returns:
        List[float]: a list of all thetas ordered 
    """    

    thetas = [2 * np.arccos(ditgamma(n, k, i, l, 0, s))]
    for m in range(1, int(2 * s)):
        val = ditgamma(n, k, i, l, m, s) / list_prod(thetas)
        val = np.clip(val, -1, 1)
        theta = 2 * np.arccos(val)
        if theta == 0: # can occur when gamma value 
            thetas.extend([0] * (int(2 * s) - len(thetas)))
            return thetas
        thetas.append(theta)
    return thetas

# ------------------------------
# Core gate operation for dit states
# ------------------------------

def ditIgate(quds:cirq.LineQid, anc_qud: cirq.NamedQid, n: int, k: int, i: int, l: int, s: float):
    """I Gate operation for dicke state creatioin

    Args:
        quds (cirq.LineQid): List of working qudits
        anc_qud (cirq.NamedQid): anciliry qudit
        n (int): number of working qudits
        k (int): number of spin operators
        i (int): the curent working qudit number
        l (int): the current I gate number
        s (float): the spin of the system
    """    

    thetas = get_dit_thetas(n, k, i, l, s)



    #General CSUM(1-x) operation 
    yield (ControlledShiftOneMinusC(int(2*s+1),k+1,-1)).on(quds[i-1],anc_qud)


    # 2s controlled rotations 
    for j in range(int(2 * s)):
        yield cirq.ControlledGate(
            ditrotation(thetas[j], j, j + 1, int(2 * s + 1)),
            num_controls=1,
            control_values=((l + 1) % (k + 1),),
            control_qid_shape=(k + 1,)
        )(anc_qud, quds[i - 1])

    #General inverse CSUM(1-x) operation
    yield (ControlledShiftOneMinusC(int(2*s+1),k+1,1)).on(quds[i-1],anc_qud)


# ------------------------------
# Full circuit assembly for dit states
# ------------------------------

def ditUgate(quds, anc_qud, n, k, i, s):
    """Indiviudal U of i. Code breaks U initary into simpler products of I gates for qudit extension

    Args:
        quds (cirq.LineQid): List of working qudits
        anc_qud (cirq.NamedQid): anciliry qudit
        n (int): number of working qudits
        k (int): number of spin operators
        i (int): the curent working qudit number
        s (float): the spin of the system

    """    
    # for l in range(int(max(0, 2*s*(i-n-1)+k)),min(int(2*s*i),k)):
    #     yield ditIgate(quds, anc_qud, n, k, i, l, s)
    for l in range(k):
        yield ditIgate(quds, anc_qud, n, k, i, l, s)

def ditU(qubs, qud, n, k, s):
    """
    Full unitary U to run U gates on each dit 

    Args:
        quds (cirq.LineQid): List of working qudits
        anc_qud (cirq.NamedQid): anciliry qudit
        n (int): number of working qudits
        k (int): number of spin operators
        s (float): the spin of the system

    """    
    for i in range(1, n + 1):
        yield ditUgate(qubs, qud, n, k, i, s)



def qudit_dicke(n,k,s):
    sim = cirq.Simulator()

    quds = cirq.LineQid.range(n, dimension=int(2 * s + 1))
    qudit = cirq.NamedQid('a', dimension=k + 1)

    circuit = cirq.Circuit(ditU(quds, qudit, n, k, s))

    print("\n=== Final State Vector ===")
    result = sim.simulate(circuit)
    #print(circuit)
    print(cirq.dirac_notation(result.final_state_vector, qid_shape=(int(2 * s + 1),) * n + (k + 1,)))



n,k,s=2,4,1

qudit_dicke(n,k,s)
