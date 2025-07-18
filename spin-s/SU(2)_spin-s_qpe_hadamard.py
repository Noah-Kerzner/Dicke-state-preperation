'''
Written by Noah Kerzner 

technical constant depth for spin-s systems using 1 ancila qudit. Further a linear superpostion scheme is written.

Read paper at https://arxiv.org/abs/2507.13308
'''


import matplotlib.pyplot as plt
import cirq
import numpy as np
from collections import Counter
from scipy.special import comb





class R_ij(cirq.Gate):
    """Generalized rotation gate between two basis states i and j in a qudit of dimension d.

    This gate performs a rotation by angle theta in the two-dimensional subspace spanned by states |i⟩ and |j⟩,
    leaving other basis states unchanged.

    Attributes:
        theta (float): angle of rotation
        d (int): dimension of the qudit
        i (int): first basis state index
        j (int): second basis state index
    """    

    def __init__(self, theta: float, d: int, i: int, j: int):
        """Initialize the rotation gate.

        Args:
            theta (float): angle of rotation in radians
            d (int): dimension of the qudit
            i (int): index of the first basis state involved in rotation
            j (int): index of the second basis state involved in rotation
        """        
        super().__init__()
        self.theta = theta
        self.d = d
        self.i = i
        self.j = j

    def _qid_shape_(self):
        """Return the shape of the qudit (dimension)."""
        return (self.d,)

    def _unitary_(self):
        """Return the unitary matrix representing the rotation.

        The unitary acts non-trivially only on the subspace spanned by |i⟩ and |j⟩,
        implementing a rotation by angle theta.
        """
        mat = np.eye(self.d, dtype=np.complex128)
        i, j = self.i, self.j
        c = np.cos(self.theta/2)
        s = np.sin(self.theta/2)
        mat[i, i] = c
        mat[j, j] = c
        mat[i, j] = -s
        mat[j, i] = s
        return mat

    def _circuit_diagram_info_(self, args):
        """Provide a string representation for circuit diagrams."""
        return f"R({self.i},{self.j},{self.theta:.2f})"





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

    def __init__(self, work_dim: float, anc_qud_dim: int):
        """
        Initialize the phase gate.

        Args:
            work_dim (float): dimension of the target qudit (d=2s+1)
            anc_qud_dim (int): dimension of the control qudit (D=2sn+1)
        """
        self.d = work_dim
        self.D = anc_qud_dim

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
        mat = np.zeros((dim, dim), dtype=np.complex128)
        for x in range(self.D):  # control index
            for y in range(self.d):  # target index
                idx = x * self.d + y
                phase = np.exp(2j * np.pi * x * y / self.D)
                mat[idx, idx] = phase
        return mat

    def _circuit_diagram_info_(self, args):
        """Labels for circuit diagrams."""
        return [f"x (D={self.D})", f"e^{{2πi·xy/{self.d}}}"]






def full_U(d:int,n:int, anc_qud: cirq.LineQid, working_vec: cirq.LineQid):
    """Apply the NumOpPhaseGate U on ancilla and each working qudit.

    Args:
        d (int): dimension of working qudits (d=2s+1)
        n (int): number of working qudits
        anc_qud (cirq.LineQid): ancillary qudit
        working_vec (list[cirq.LineQid]): list of working qudits
    Yields:
        cirq.Operation: operations implementing U on ancilla and each working qudit
    """
    
    U=NumOpPhaseGate(d,(d-1)*n+1)

    for i in range(n):
        # Apply U on ancilla and i-th working qudit
        yield U.on(anc_qud,working_vec[i])








def qpe(k:int, dim_anc: int, d: int,n:int,  working_vec: list[cirq.Qid], mode=0):
    """Quantum Phase Estimation (QPE) circuit for spin-s systems.

    This function constructs the QPE circuit using one ancilla qudit and multiple working qudits.

    Args:
        k (int): parameter related to the number of spin operations applied
        dim_anc (int): dimension of the ancilla qudit
        d (int): dimension of working qudits (d=2s+1)
        n (int): number of working qudits
        working_vec (list[cirq.Qid]): list of working qudits
        mode (int): 0 to include measurement, 1 to skip measurement (default 0)
    Yields:
        cirq.Operation: operations forming the QPE circuit
    """

    anc_qud=cirq.LineQid(0,dimension=dim_anc)

    # Apply Hadamard (Fourier) to ancilla qudit to create superposition
    yield qudH(dim_anc).on(anc_qud)

    # Apply controlled-U operations
    yield full_U(d,n,anc_qud,working_vec)

    # Apply inverse Hadamard (inverse Fourier) to ancilla qudit
    yield qudH(dim_anc).inverse().on(anc_qud) 


    if mode==0: # mode to measure or not 
        # Measure ancilla qudit to estimate phase (k)
        yield cirq.measure(anc_qud, key='k_estimate')
        
        # Measure working qudits (Dicke state)
        yield cirq.measure(working_vec,key='dicke')




def ditgamma(s,k,n,m):
    """Calculate the gamma factor used in the construction of qdit Dicke states.

    This factor corresponds to the amplitude coefficient for the m-th basis state of a working qudit,
    given parameters s (spin), k (number of spin operations), and n (number of qudits).

    Args:
        s (float): spin of the system (spin-s)
        k (int): number of spin operations applied
        n (int): number of working qudits
        m (int): basis state index of the working qudit (0 ≤ m ≤ 2s)

    Returns:
        float: gamma coefficient amplitude for the m-th basis state
    """    
    
    p=k/(2*s*n)  # probability-like parameter related to k, s, and n
    val=(1-p)**s
    val*=np.sqrt(comb(2*s,m))  # binomial coefficient sqrt
    val*=(p/(1-p))**(m/2)  # weighting factor for basis state m
    return val



def get_dit_thetas(n, k, s):
    """Compute a list of rotation angles (thetas) for preparing the Dicke state on qudits.

    These angles parameterize rotations R_ij that generate the Dicke state amplitudes.

    Args:
        n (int): number of working qudits
        k (int): number of spin operations applied
        s (float): spin of the system (spin-s)

    Returns:
        list: list of rotation angles theta_m for m=0 to 2s-1
    """
    thetas = []
    epsilon = 1e-10

    # Compute the first angle theta_1 from gamma factor for m=0
    theta_1 = 2 * np.arccos(np.clip(ditgamma(s, k, n, 0), -1, 1))
    thetas.append(theta_1)

    # For subsequent m, compute angles using recursive relation involving products of sines of previous thetas
    for m in range(1, int(2 * s)):
        denom = list_prod(thetas)
        if abs(denom) < epsilon:
            # If denominator too small, fill remaining angles with zeros to avoid numerical issues
            thetas.extend([0 for _ in range(int(2 * s - len(thetas)))])
            return thetas

        bin_expr = ditgamma(s, k, n, m) / denom
        bin_expr = np.clip(bin_expr, -1, 1)
        theta_m = 2 * np.arccos(bin_expr)

        if np.isnan(theta_m):  # safety check for numerical errors
            thetas.extend([0 for _ in range(int(2 * s - len(thetas)))])
            return thetas

        thetas.append(theta_m)

    return thetas


def list_prod(arr):
    """Compute the product of sin(theta/2) for each angle theta in the list.

    Args:
        arr (list): list of angles in radians

    Returns:
        float: product of sin(theta/2) values
    """
    ret=1

    for arg in arr:
        ret*=np.sin(arg/2)
    return ret





def dicke_simulate(n:int,k: int, s: float,print_circ=False):
    """Simulate the QPE algorithm for the Dicke state without measurement.

    Constructs and simulates the circuit preparing the Dicke state and applying QPE,
    outputting the final state vector before measurement.

    Args:
        n (int): number of working qudits
        k (int): desired number of spin operations (related to Dicke state)
        s (float): spin of the system (spin-s)
    """
    d=int(2*s+1)  # dimension of working qudits

    dim_anc=int(2*s*n+1)  # dimension of ancilla qudit

    # Prepare working qudits with labels starting at l (which is undefined here, assuming 0)
    l=1  # added definition for l to avoid error
    quds=[]
    for i in range(n):
        quds.append(cirq.LineQid(l+i,dimension=d))

    circuit=cirq.Circuit()

    # Calculate rotation angles for Dicke state preparation
    thetas=get_dit_thetas(n,k,s)

    # Apply rotations R_ij on each working qudit to prepare Dicke state
    for i,theta in enumerate(thetas):  
        R=R_ij(theta,d,i,i+1)
        for qud in quds: 
            circuit.append(R.on(qud))

    # Append QPE circuit without measurement
    circuit.append(qpe(k,dim_anc,d,n,quds,1))

    # Simulate the circuit
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    if print_circ:
        print(circuit)
    # Print the final state vector in Dirac notation
    print(cirq.dirac_notation(result.final_state_vector, qid_shape=(dim_anc,) + (d,)*n))
   


def dicke_shots(n,k,s,shots=10,mode=0): 
    """Run repeated QPE algorithm shots to generate measurement statistics and plot results.

    This function runs the QPE circuit multiple times, collects measurement outcomes,
    and plots histograms of the final state distributions.

    Args:
        n (int): number of working qudits
        k (int): desired number of spin operations (related to Dicke state)
        s (float): spin of the system (spin-s)
        shots (int, optional): number of repetitions (default 10)
        mode (int, optional): 0 for full distribution, 1 for distribution conditioned on expected k (default 0)
    """    
    
    d=int(2*s+1)  # dimension of working qudits

    # Number of qubits needed to encode ancilla dimension (ceil of log base 2)
    l = int(np.ceil(np.log((d-1)*n + 1)/np.log(2)))

    # Prepare working qudits starting at label l
    quds=[]
    for i in range(n):
        quds.append(cirq.LineQid(l+i,dimension=d))

    circuit=cirq.Circuit()

    # Calculate rotation angles for Dicke state preparation
    thetas=get_dit_thetas(n,k,s)

    # Apply rotations R_ij on each working qudit
    for i,theta in enumerate(thetas):  
        R=R_ij(theta,d,i,i+1)
        for qud in quds: 
            circuit.append(R.on(qud))

    # Append QPE circuit with measurement
    circuit.append(qpe(k,l,d,n,quds,0))

    # Simulate circuit with repeated shots
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=shots)

    # Extract bitstrings of ancilla measurement results
    bitstrings = ["".join(str(bit) for bit in bits) for bits in result.measurements['k_estimate']]
    
    counts = Counter(bitstrings)
    most_common_k = counts.most_common(1)[0][0]
    #print(counts)
    estimated_k = int(most_common_k, 2)
    dicke_bits = result.measurements['dicke']

    if mode == 1:
        # Filter measurement results to only those with digit sum equal to k
        dickes_meas = [
        "".join(str(bit) for bit in bits)
        for bits in dicke_bits
        if sum(int(b) for b in bits) == k  # keep only bitstrings where the digit sum equals k
    ]
    else:
        # Use all measurement results
        dickes_meas = ["".join(str(bit) for bit in bits) for bits in dicke_bits]

    dickes=Counter(dickes_meas)
    analyze_counters(dickes, estimated_k)




def analyze_counters( dickes: Counter, estimated_k: int):
    """Plot a bar graph of measurement results from Dicke state simulations.

    Args:
        dickes (Counter): counted measurement results of final state vectors
        estimated_k (int): most probable k value from measurement
    """    
    # Plot histogram from measurement counts
    labels, values = zip(*dickes.items())
    plt.bar(labels, values)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(f'Distribution of final state vectors for k ≈ {estimated_k}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()



if __name__=="__main__":
    # Run linear Dicke state simulation with parameters n=3, k1=2, k2=1, s=1
    #dicke_shots(2,2,1,1000)  # example usage commented out
    dicke_simulate(2,4,3,True) # state output has ancila as first qudit then working


