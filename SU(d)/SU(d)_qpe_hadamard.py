# Read paper at https://arxiv.org/abs/2507.13308


import cirq
import numpy as np









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



class qudH(cirq.Gate):
    """Generalized Hadamard gate for qudits of dimension d.

    This gate implements the quantum Fourier transform over Z_d,
    mapping computational basis states to equal superpositions with phase factors.

    Attributes:
        dim (int): dimension of the qudit
    """

    def __init__(self, dim):
        self.dim = dim

    def _qid_shape_(self):
        """Return the qudit dimension."""
        return (self.dim,)

    def _unitary_(self):
        """Return the unitary matrix for the qudit Hadamard (quantum Fourier transform).

        The matrix elements are ω^{jk} / sqrt(d), with ω = exp(2πi/d).
        """
        omega = np.exp(2j * np.pi / self.dim)
        return np.array([[omega ** (j * k) / np.sqrt(self.dim) for k in range(self.dim)] for j in range(self.dim)])

    def _circuit_diagram_info_(self, args):
        """Return the label for circuit diagrams."""
        return f"H_d{self.dim}"

    def inverse(self):
        """Return the inverse of the qudit Hadamard gate (inverse Fourier transform)."""
        return cirq.MatrixGate(np.conj(self._unitary_().T), name='H_dag', qid_shape=(self.dim,))



class NumOpPhaseGate(cirq.Gate):
    """Phase gate implementing the unitary U|x⟩|y⟩ = e^{2πi * x * y / D} |x⟩|y⟩.

    This gate applies a phase depending on the product of control x and target y indices.

    Attributes:
        d (int): dimension of the target qudit (d=2s+1)
        D (int): dimension of the control qudit (D=2sn+1)
    """

    def __init__(self, work_dim: int, anc_qud_dim: int, i: int):
        """
        Initialize the phase gate.

        Args:
            work_dim (float): dimension of the target qudit (d=2s+1)
            anc_qud_dim (int): dimension of the control qudit (D=2sn+1)
        """
        self.d = work_dim
        self.D = anc_qud_dim
        self.i=i

    def _num_qubits_(self):
        """Return the number of qubits (always 2 in this context)."""
        return 2

    def _qid_shape_(self):
        """Return the shape of control and target qudits."""
        return (self.D, self.d)

    def _unitary_(self):
        """Return the unitary matrix representing the phase gate.

        The diagonal elements are e^{2πi * x * y / D} for indices x (control) and y (target).
        """
        dim = self.D * self.d
        mat = np.eye(dim, dtype=complex)
        for x in range(self.D):
            idx = x * self.d + self.i  # index of |x⟩|i⟩
            mat[idx, idx] = np.exp(2j * np.pi * x / self.D)
        return mat

    def _circuit_diagram_info_(self, args):
        """Labels for circuit diagrams."""
        return [f"x (D={self.D})", f"e^{{2πi·x/{self.D}}}"]








def h(i: int, d: int) -> np.ndarray:
    """
    Returns the projector |i><i| for a qudit of dimension d.

    Args:
        i (int): Index of the basis state (0 ≤ i < d).
        d (int): Dimension of the qudit Hilbert space.

    Returns:
        np.ndarray: A d x d matrix representing the projector |i><i|.
    """
    if not (0 <= i < d):
        raise ValueError("Index i must be in the range 0 ≤ i < d")
    
    vec = np.zeros((d, 1), dtype=complex)
    vec[i, 0] = 1.0
    return vec @ vec.conj().T  # |i⟩⟨i|








def full_U(d:int,n:int, anc_qubs: list[cirq.LineQubit], working_vec: list[cirq.LineQid]):

    
   


    for p, qud in enumerate(anc_qubs):
        
        U=NumOpPhaseGate(d,n+1,p+1)

        for i in range(len(working_vec)):

            
            yield U.on(qud,working_vec[i])








def qpe(num_qubs: int, d: int,n:int,  working_vec: list[cirq.Qid], mode=0):


    anc_qubs=[]


    for i in range(d-1):


        anc_qubs.append(cirq.LineQid(i,n+1))


    
    # Apply Hadamard (Fourier) to ancilla qudit to create superposition
    for qud in anc_qubs:
        yield qudH(n+1).on(qud) 

    # Apply controlled-U operations
    yield full_U(d,n,anc_qubs,working_vec)

    # Apply inverse Hadamard (inverse Fourier) to ancilla qudit
    for qud in anc_qubs:
        yield qudH(n+1).inverse().on(qud)
        pass 

    # Apply cyclic shift down by k to ancilla qudit
    #yield ditminus(dim_anc-1,k).on(anc_qud)



    if mode==0: # mode to measure or not 
        # Measure

        for i,qub in enumerate(anc_qubs):
            yield cirq.measure(qub, key=f'k_{i+1}_estimate')
        
        yield cirq.measure(working_vec,key='dicke')














def get_dit_thetas(n:int ,k:np.ndarray):
        thetas = []

        zs=[np.sqrt(k[i]/n) for i in range(len(k)-1)]

        for z in  zs:

            if z == 0:
                thetas.append(np.pi)
                continue

            denom = list_prod(thetas)
            
            if denom < 1e-10:
                thetas.append(0)
                continue

            val = z/ denom
            val = np.clip(val, -1, 1)
            theta = 2 * np.arccos(val)

            thetas.append(theta)

        #print(f'gamms/thetas for i= {i}, p= {p}\n{gamms}\n{thetas}\n')
        return thetas




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




def init_qud(quds:cirq.LineQid, k_vec,n,d):

    thetas=get_dit_thetas(n,k_vec)


    for i,theta in enumerate(thetas):
        for qud in quds:
            yield ditrotation(theta,i,i+1,d).on(qud)

    






def dicke_simulate(k: tuple,print_circ=False):
    """
    runs the qpe algo for the dicke state but does not measure.

    - The function ouptuts the final state pre measurment of the total system
    - the order is ancilary qubits then working qubits

    Args:
        n (int): number of working qubits
        k (int): desired number of 1's in dicke state
    """


    n=sum(k)
    d=len(k)

    l=d-1


    quds=[]
    for i in range(n):
        quds.append(cirq.LineQid(l+i,dimension=d))

    circuit=cirq.Circuit()


    circuit.append(init_qud(quds,k,n,d))

    circuit.append(qpe(l,d,n,quds,1))

    # Simulate
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    if print_circ:
        print(circuit)
    print(cirq.dirac_notation(result.final_state_vector, qid_shape=(n+1,) * (d-1) + (d,)*n))







k=(1,1,2)
dicke_simulate(k)

