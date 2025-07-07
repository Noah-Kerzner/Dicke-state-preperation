import cirq
import numpy as np
from scipy.special import comb






class CShiftUpGate(cirq.Gate):
    def __init__(self, d_control, d_target):
        self.d_control = d_control
        self.d_target = d_target

    def _num_qubits_(self):
        return 2

    def _qid_shape_(self):
        return (self.d_control, self.d_target)  # control, target qudits of dimension d_control and d_target

    def _unitary_(self):
        d_control = self.d_control
        d_target = self.d_target
        dim = d_control * d_target
        U = np.zeros((dim, dim), dtype=complex)

        for x in range(d_control):         # control value
            for y in range(d_target):     # target value
                input_index = x * d_target + y
                output_y = (y + x) % d_target
                output_index = x * d_target + output_y
                U[output_index, input_index] = 1

        return U

    def _circuit_diagram_info_(self, args):
        return [f"C(d={self.d_control})", f"ShiftUp(d={self.d_target})"]






class ControlledDitRotation(cirq.Gate):
    def __init__(self, thetas: list[float], i: int, j: int, d_control: int, d_target: int):
        """
        Args:
            thetas (List[float]): List of angles for each control qudit value.
            i (int): Index of the first basis state for the rotation on the target.
            j (int): Index of the second basis state for the rotation on the target.
            d_control (int): Dimension of the control qudit.
            d_target (int): Dimension of the target qudit.
        """
        if i == j or not (0 <= i < d_target) or not (0 <= j < d_target):
            raise ValueError("Indices i and j must be different and in range [0, d_target-1]")
        if len(thetas) != d_control:
            raise ValueError("Length of theta list must match dimension of control qudit.")
        self.thetas = thetas
        self.i = i
        self.j = j
        self.d_control = d_control
        self.d_target = d_target

    def _num_qubits_(self):
        return 2

    def _qid_shape_(self):
        return (self.d_control, self.d_target)

    def _unitary_(self):
        dim = self.d_control * self.d_target
        U = np.eye(dim, dtype=np.complex128)
        for x in range(self.d_control):
            theta = self.thetas[x]
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            # Submatrix for rotation
            for a, b, val in [(self.i, self.i, c), (self.j, self.j, c),
                              (self.i, self.j, -s), (self.j, self.i, s)]:
                row = x * self.d_target + a
                col = x * self.d_target + b
                U[row, col] = val
        return U

    def _circuit_diagram_info_(self, args):
        return [f"C", f"R{self.i}{self.j}(θ[x])"]





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
        prod_val = list_prod(thetas)
        if prod_val == 0:
            thetas.extend([0] * (int(2 * s) - len(thetas)))
            return thetas
        val = ditgamma(n, k, i, l, m, s) / prod_val
        val = np.clip(val, -1, 1)
        theta = 2 * np.arccos(val)
        if theta == 0 or np.isnan(theta): # can occur when gamma value 
            thetas.extend([0] * (int(2 * s) - len(thetas)))
            return thetas
        thetas.append(theta)
    return thetas




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

    thetas=[]
    for l in range(k+1):
        thetas.append(get_dit_thetas(n, k, i, l, s))
    thetas=np.array(thetas)



    for j in range(int(2*s)):
        yield ControlledDitRotation(thetas[:,j],j,j+1,k+1,int(2*s+1)).on(anc_qud,quds[i-1])

    yield CShiftUpGate(int(2*s+1),k+1).on(quds[i-1],anc_qud)


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
    print(circuit)
    print(cirq.dirac_notation(result.final_state_vector, qid_shape=(int(2 * s + 1),) * n + (k + 1,)))



n,k,s=3,2,1

qudit_dicke(n,k,s)