import matplotlib.pyplot as plt
import cirq
import numpy as np
from collections import Counter
from scipy.special import comb
from functools import reduce
from scipy.linalg import expm
from quantum_forier_transform import qud_iqft, qudH


class DickeGate(cirq.Gate):
    def __init__(self, matrix,d,n):
        super().__init__()
        self._matrix = matrix
        self.d=d
        self.n=n

    def _num_qubits_(self):
        return self.n

    def _unitary_(self):
        return self._matrix

    def __pow__(self, power):
        if isinstance(power, int):
            return DickeGate(np.linalg.matrix_power(self._matrix, power),self.d,self.n)
        else:
            return NotImplemented
        
    def _qid_shape_(self):
        n = int(np.log(self._matrix.shape[0]) / np.log(self.d))
        return (self.d,) * n
        
    def _circuit_diagram_info_(self, args):
        n = self._num_qubits_()
        return cirq.CircuitDiagramInfo(
            wire_symbols=[f'U_dicke'] + [''] * (n - 1),
            exponent=1,
            connected=True
        )


# Controlled Dicke gate for qudits
class qudCDicke(cirq.Gate):
    def __init__(self, dim, dicke_unitary, exponent=1.0):
        self.dim = dim
        self.dicke_unitary = dicke_unitary
        self.exponent = exponent

    def _qid_shape_(self):
        num_dicke_qudits = int(np.log(self.dicke_unitary.shape[0]) / np.log(self.dim))
        return (self.dim,) + (self.dim,) * num_dicke_qudits

    def _unitary_(self):
        d = self.dim
        n = int(np.log(self.dicke_unitary.shape[0]) / np.log(d))
        total_dim = d ** (n + 1)
        mat = np.eye(total_dim, dtype=complex)

        for ctrl_val in range(d):
            block_start = ctrl_val * d**n
            block_end = block_start + d**n
            mat[block_start:block_end, block_start:block_end] = (
                mat[block_start:block_end, block_start:block_end] @
                np.linalg.matrix_power(self.dicke_unitary, int(self.exponent * ctrl_val))
            )

        return mat

    def _circuit_diagram_info_(self, args):
        return [f"@_d{self.dim}", "U_dicke"] + [""] * (len(self._qid_shape_()) - 2)

    def __pow__(self, exponent):
        return qudCDicke(self.dim, self.dicke_unitary, self.exponent * exponent)
    





def qpe(U: cirq.Gate, num_qubs: int, d: int, working_vec: list[cirq.Qid], mode=0, qubits=True):


    if qubits:
        #anc_quds=cirq.LineQid.range(num_qubs, dimension=2)
        anc_quds = cirq.LineQubit.range(num_qubs)

        
        # Apply Hadamards to ancillas
        for qud in anc_quds:
            yield cirq.H.on(qud)

        #Apply controlled-U^{d^i}
        for i, qub in enumerate(anc_quds):
            yield cirq.ControlledGate(
                U**(2**(num_qubs - i - 1)),
                control_values=[1],
                control_qid_shape=(2,)
            ).on(qub, *working_vec)

        yield cirq.qft(*anc_quds, inverse=True)

    else:

        anc_quds=cirq.LineQid.range(num_qubs, dimension=d)
        

        H=qudH(d)
        # Apply Hadamards to ancillas
        for qud in anc_quds:
            yield H.on(qud)

        # Apply controlled-U^{d^i}
        for i, qub in enumerate(anc_quds):
            powered_U = U**(d**(num_qubs - i - 1))
            yield qudCDicke(d, powered_U._unitary_()).on(qub, *working_vec)


        # Inverse QFT
        yield qud_iqft(anc_quds,d)
        


    if mode==0: # mode to measure or not 
        # Measure
        yield cirq.measure(*anc_quds, key='k_estimate')
        
        yield cirq.measure(working_vec,key='dicke')






def dicke_unitary(n: int,d:int) -> tuple[cirq.Gate, int]:
    """form the dicke state creation unitary operator for qpe

    Args:
        n (int): number of qudits in dicke state
        d (int)L dimensions of qudits
    Returns:
        cirq.Gate: unitary gate operator to be used in qpe
    """
    blank_n = np.array([np.identity(d)] * n)
    num_op = np.array(np.diag([i for i in range(d)]))

    N = np.zeros((d**n, d**n))
    for i in range(n):
        blank_n[i] = num_op
        N += kron_product(blank_n)
        blank_n[i] = np.identity(d)

    
    l = int(np.ceil(np.log((d-1)*n + 1)/np.log(2)))
    matrix = expm(2j * np.pi * N / (2**l))

    return DickeGate(matrix,d,n), l

def kron_product(matrices: np.ndarray) -> np.ndarray:
    """
        kronecker product of a np.array of matrices. kronecker prodcut goes from indexes 0 to n-1
    Returns:
        np.ndarray: resulting kronecker product of the list
    """
    return reduce(np.kron, matrices)


def ditgamma(s,k,n,m):
    """gamma factor for qdit dicke state 

    Args:
        n (int): num qdits
        k (int): amount of spin ops applied 
        l (int): current ancilery qudit value/ I gate index 
        m (int): value of working qudit (not anciliry)
        s (float): spin of system

    Returns:
        float: gamma value 
    """    
    
    p=k/(2*s*n)
    val=(1-p)**s
    val*=np.sqrt(comb(2*s,m))
    val*=(p/(1-p))**(m/2)
    return val



def get_dit_thetas(n, k, s):
    thetas = []
    epsilon = 1e-10

    theta_1 = 2 * np.arccos(np.clip(ditgamma(s, k, n, 0), -1, 1))
    thetas.append(theta_1)

    for m in range(1, int(2 * s)):
        denom = list_prod(thetas)
        if abs(denom) < epsilon:
            thetas.extend([0 for _ in range(int(2 * s - len(thetas)))])
            return thetas

        bin_expr = ditgamma(s, k, n, m) / denom
        bin_expr = np.clip(bin_expr, -1, 1)
        theta_m = 2 * np.arccos(bin_expr)

        if np.isnan(theta_m):  # safety check
            thetas.extend([0 for _ in range(int(2 * s - len(thetas)))])
            return thetas

        thetas.append(theta_m)

    return thetas


def list_prod(arr):
    ret=1

    for arg in arr:
        ret*=np.sin(arg/2)
    return ret


def dicke_simulate(n:int,k:int,s:int):
    """
    runs the qpe algo for the dicke state but does not measure.

    - The function ouptuts the final state pre measurment of the total system
    - the order is ancilary qubits then working qubits

    Args:
        n (int): number of working qubits
        k (int): desired number of 1's in dicke state
    """
    d=int(2*s+1)
    U,l= dicke_unitary(n,d)


    quds=[]
    for i in range(n):
        quds.append(cirq.LineQid(l+i,dimension=d))

    circuit=cirq.Circuit()

    H=qudH(d)
    for qud in quds: # initializes proper eigenvector
        circuit.append(H.on(qud))

    circuit.append(qpe(U,l,d,quds,1))

    # Simulate
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    #print(circuit)
    print(cirq.dirac_notation(result.final_state_vector, qid_shape=(2,) * (l) + (d,)*n))



def dicke_shots(n,k,s,shots=10,mode=0): # mode=0 for full distrobution, mode = 1 for expected k distrobution
    """runs repeated qpe alogirthm to make bar graph of final vector measurment and l

    - Mode = 0 prints a bar graph of the total probability distrobution (all k's and all state vectors)
    - Mode = 1 prints a bar graph of the probabiliyt distrobution only for the measurments in the expected k

    Args:
        n (_type_): amount of working qubits
        k (_type_): desired number of 1's 
        shots (int, optional): amount of instances of the QPE algorithm. Defaults to 10.
        mode (int, optional): mode of operation. Defaults to 0.
    """    
    
    d=int(2*s+1)
    U,l= dicke_unitary(n,d)


    quds=[]
    for i in range(n):
        quds.append(cirq.LineQid(l+i,dimension=d))

    circuit=cirq.Circuit()

    H=qudH(d)
    for qud in quds: # initializes proper eigenvector
        circuit.append(H.on(qud))

    circuit.append(qpe(U,l,d,quds,0))

    # Simulate
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=shots)


    bitstrings = ["".join(str(bit) for bit in bits) for bits in result.measurements['k_estimate']]
    
    counts = Counter(bitstrings)
    most_common_k = counts.most_common(1)[0][0]
    print(counts)
    estimated_k = int(most_common_k, d)
    dicke_bits = result.measurements['dicke']
    if mode == 1:
        dickes_meas = [
        "".join(str(bit) for bit in bits)
        for bits in dicke_bits
        if sum(int(b) for b in bits) == k  # keep only bitstrings where the digit sum equals k
    ]
    else:
        dickes_meas = ["".join(str(bit) for bit in bits) for bits in dicke_bits]

    dickes=Counter(dickes_meas)
    analyze_counters(dickes, estimated_k)






def analyze_counters( dickes: Counter, estimated_k: int):
    """Function to make bar graph of measurment results

    Args:
        counter2 (Counter): counted and ordered dicke measrument results 
        estimated_k (int): most probable k from measurment
    """    
    # Print most common value from counter1

    #print(estimated_k, 'k val')
    # Plot histogram from counter2
    labels, values = zip(*dickes.items())
    plt.bar(labels, values)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(f'Distribution of final state vectors for k ≈ {estimated_k}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()




def test_qudit_qpe():
    d = 3  # Qudit dimension
    n = 5  # Number of ancilla qudits
    phase = 10/ d**n  # Set known phase

    # Define eigenvector |1⟩ (eigenstate of Z_d)
    working_qudit = cirq.LineQid(n,d)
    

    # Define the phase gate U = diag(1, ω, ω^2)
    omega = np.exp(2j * np.pi / d)
    diag = np.array([np.exp(2j * np.pi * i * phase) for i in range(d)], dtype=complex)
    U = cirq.MatrixGate(np.diag(diag), qid_shape=(d,))

    # Build circuit
    circuit = cirq.Circuit()
    circuit.append(cirq.MatrixGate(np.roll(np.eye(d), 1, axis=1), qid_shape=(d,)).on(working_qudit))  # Apply X-like gate to set to |1⟩
    circuit.append(qpe(U, num_qubs=n, d=d, working_vec=working_qudit, mode=0))

    # Run multiple repetitions
    sim = cirq.Simulator()
    reps = 1000
    results = sim.run(circuit, repetitions=reps)

    # Extract and decode measurement results
    counts = Counter("".join(str(bit) for bit in row) for row in results.measurements['k_estimate'])
    print(counts)
    most_common_k_str, _ = counts.most_common(1)[0]
    estimated_k = int(most_common_k_str, d)
    estimated_phi = estimated_k / d**n

    print(f"Most common measurement (base-{d}): {most_common_k_str}")
    print(f"Estimated phase φ ≈ {estimated_phi}")
    print(f"True phase φ = {phase}")


#dicke_shots(2,2,1,1000,0)


class R_ij(cirq.Gate):
    def __init__(self, theta: float, d: int, i: int, j: int):
        super().__init__()
        self.theta = theta
        self.d = d
        self.i = i
        self.j = j

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
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
        return f"R({self.i},{self.j},{self.theta:.2f})"




def boost_dicke_shots(n,k,s,shots=10,mode=0,qubits=True): # mode=0 for full distrobution, mode = 1 for expected k distrobution
    """runs repeated qpe alogirthm to make bar graph of final vector measurment and l

    - Mode = 0 prints a bar graph of the total probability distrobution (all k's and all state vectors)
    - Mode = 1 prints a bar graph of the probabiliyt distrobution only for the measurments in the expected k

    Args:
        n (_type_): amount of working qubits
        k (_type_): desired number of 1's 
        shots (int, optional): amount of instances of the QPE algorithm. Defaults to 10.
        mode (int, optional): mode of operation. Defaults to 0.
    """    
    
    d=int(2*s+1)
    U,l= dicke_unitary(n,d)


    quds=[]
    for i in range(n):
        quds.append(cirq.LineQid(l+i,dimension=d))

    circuit=cirq.Circuit()

    thetas=get_dit_thetas(n,k,s)
    print(thetas)
    for i,theta in enumerate(thetas):  
        R=R_ij(theta,d,i,i+1)
        for qud in quds: 
            circuit.append(R.on(qud))
        

    circuit.append(qpe(U,l,d,quds,0,qubits))

    # Simulate
    simulator = cirq.Simulator()
    # result=simulator.simulate(circuit)
    # print(cirq.dirac_notation(result.final_state_vector, qid_shape=(2,) * (l) + (d,)*n))
    result = simulator.run(circuit, repetitions=shots)


    bitstrings = ["".join(str(bit) for bit in bits) for bits in result.measurements['k_estimate']]
    
    counts = Counter(bitstrings)
    most_common_k = counts.most_common(1)[0][0]
    print(counts.most_common(5))
    estimated_k = int(most_common_k, 2) if qubits else int(most_common_k, d)
    dicke_bits = result.measurements['dicke']
    if mode == 1:
        dickes_meas = [
        "".join(str(bit) for bit in bits)
        for bits in dicke_bits
        if sum(int(b) for b in bits) == k  # keep only bitstrings where the digit sum equals k
    ]
    else:
        dickes_meas = ["".join(str(bit) for bit in bits) for bits in dicke_bits]

    dickes=Counter(dickes_meas)
    analyze_counters(dickes, estimated_k)





#dicke_shots(2,2,1,1000) 
#dicke_simulate(2,2,1)
# print('\n')
#dicke_boosted_shots(2,2,1,1000)
#test_boosted_initial_state_dirac()

#test_boosted_distribution()


def test_init(n,k,s):
    d=int(2*s+1)


    quds=[]
    for i in range(n):
        quds.append(cirq.LineQid(i,dimension=d))

    circuit=cirq.Circuit()

    thetas=get_dit_thetas(n,k,s)

    for i,theta in enumerate(thetas):  
        R=R_ij(theta,d,i,i+1)
        for qud in quds: 
            circuit.append(R.on(qud))
        
    
    


    # Simulate
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    print(cirq.dirac_notation(result.final_state_vector, qid_shape=(d,)*n),'\n')


    p=k/(2*s*n)
    vals=np.array([ditgamma(s,k,n,m) for m in range(d)])
    print(vals,'\n')
    vecs=np.array([vals]*n)
    vecs=kron_product(vecs)
    print(vecs)



import sys

#test_init(1,1,3/2)
# matrix=dicke_unitary(2,3)
# eigvals, eigvecs = np.linalg.eig(matrix)
# print("Eigenvalues:\n", eigvals)
# print("\nEigenvectors (columns):\n", eigvecs)
# k = int(sys.argv[1]) 
boost_dicke_shots(2, 2, 1, 1000, 0, True)