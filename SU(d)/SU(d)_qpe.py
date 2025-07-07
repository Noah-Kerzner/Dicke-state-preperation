import matplotlib.pyplot as plt
import cirq
import numpy as np
from collections import Counter
from functools import reduce
from scipy.linalg import expm




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







def full_U(i:int,l:int,d:int,n:int, anc_qubs: list[cirq.LineQubit], working_vec: cirq.LineQid):
    
    h_k=h(i+1,d)

    for p, qub in enumerate(anc_qubs[i]):

        for k in range(n):

            

            U=expm(2j * np.pi * h_k / (2**l)*(2**(l - p - 1)))
            
            
            yield cirq.ControlledGate(
                cirq.MatrixGate(U,name=f"U_{i}^{(2**(l - p - 1))}",qid_shape=(d,)),
                control_values=(1,),
                control_qid_shape=(2,)
            ).on(qub, working_vec[k])








def qpe(num_qubs: int, d: int,n:int,  working_vec: list[cirq.Qid], mode=0):


    anc_qubs=[]


    for i in range(d-1):

        temp=[]
        for j in range(num_qubs):
            temp.append(cirq.LineQubit(i*num_qubs+j))

        anc_qubs.append(temp)


    
    # Apply Hadamards to ancillas
    for quds in anc_qubs:
        for qud in quds:
            yield cirq.H.on(qud)

    #Apply controlled-U^{d^i}
    for i in range(d-1):
        yield full_U(i,num_qubs,d,n,anc_qubs,working_vec)

    for qubs in anc_qubs:

        yield cirq.qft(*qubs, inverse=True)


    if mode==0: # mode to measure or not 
        # Measure

        for i,qubs in enumerate(anc_qubs):
            yield cirq.measure(*qubs, key=f'k_{i+1}_estimate')
        
        yield cirq.measure(working_vec,key='dicke')











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

    






def dicke_simulate(k: tuple):
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

    l=int(np.ceil(np.log(n + 1)/np.log(2)))


    quds=[]
    for i in range(n):
        quds.append(cirq.LineQid((d-1)*l+i,dimension=d))

    circuit=cirq.Circuit()


    circuit.append(init_qud(quds,k,n,d))

    circuit.append(qpe(l,d,n,quds,1))

    # Simulate
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    print(circuit)
    print(cirq.dirac_notation(result.final_state_vector, qid_shape=(2,) * (l*(d-1)) + (d,)*n))


def dicke_shots(k:tuple,shots=10,mode=0): # mode=0 for full distrobution, mode = 1 for expected k distrobution
    """runs repeated qpe alogirthm to make bar graph of final vector measurment and l

    - Mode = 0 prints a bar graph of the total probability distrobution (all k's and all state vectors)
    - Mode = 1 prints a bar graph of the probabiliyt distrobution only for the measurments in the expected k

    Args:
        k (tuple): k vector for set
        shots (int, optional): amount of instances of the QPE algorithm. Defaults to 10.
        mode (int, optional): mode of operation. Defaults to 0.
    """    
    
    n=sum(k)
    d=len(k)

    l=int(np.ceil(np.log(n + 1)/np.log(2)))


    quds=[]
    for i in range(n):
        quds.append(cirq.LineQid((d-1)*l+i,dimension=d))

    circuit=cirq.Circuit()


    circuit.append(init_qud(quds,k,n,d))

    circuit.append(qpe(l,d,n,quds,0))

    # Simulate
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=shots)


    bitstrings = [["".join(str(bit) for bit in bits) for bits in result.measurements[f'k_{i+1}_estimate']] for i in range(d-1)]
    
    counts = [Counter(bits) for bits in bitstrings]
    most_common_ks = [count.most_common(1)[0][0] for count in counts]
    estimated_ks = [int(most_common_k, 2) for most_common_k in most_common_ks]
    estimated_ks.insert(0,n-sum(estimated_ks))

    dicke_bits = result.measurements['dicke']
    if mode == 1:
        dickes_meas = [
        "".join(str(bit) for bit in bits)
        for bits in dicke_bits
        if sum(bits) == k  # keep only bitstrings with exactly k ones
    ]
    else:
        dickes_meas = ["".join(str(bit) for bit in bits) for bits in dicke_bits]

    dickes=Counter(dickes_meas)
    analyze_counters(dickes, estimated_ks)






def analyze_counters( dickes: Counter, estimated_k: int):
    """Function to make bar graph of measurment results

    Args:
        counter2 (Counter): counted and ordered dicke measrument results 
        estimated_k (int): most probable k from measurment
    """    
    # Print most common value from counter1

    #print(estimated_k)
    # Plot histogram from counter2
    labels, values = zip(*dickes.items())
    plt.bar(labels, values)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(f'Distribution of final state vectors for k ≈ {estimated_k}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()





k=(1,0,0,1)
dicke_simulate(k)
#dicke_shots(k,100)