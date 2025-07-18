# Read paper at https://arxiv.org/abs/2507.13308

import matplotlib.pyplot as plt
import cirq
import numpy as np
from collections import Counter
from scipy.special import comb
from scipy.linalg import expm






class CShiftUpGate(cirq.Gate):
    def __init__(self, d):
        self.d = d

    def _num_qubits_(self):
        return 2

    def _qid_shape_(self):
        return (self.d, self.d)  # control, target qudits of dimension d

    def _unitary_(self):
        d = self.d
        dim = d * d
        U = np.zeros((dim, dim), dtype=complex)

        for x in range(d):         # control value
            for y in range(d):     # target value
                input_index = x * d + y
                output_y = (y + x) % d
                output_index = x * d + output_y
                U[output_index, input_index] = 1

        return U

    def _circuit_diagram_info_(self, args):
        return [f"C", f"ShiftUp(d={self.d})"]







class CShiftDownGate(cirq.Gate):
    def __init__(self, d):
        self.d = d

    def _num_qubits_(self):
        return 2

    def _qid_shape_(self):
        return (self.d, self.d)

    def _unitary_(self):
        d = self.d
        dim = d * d
        U = np.zeros((dim, dim), dtype=complex)
        for x in range(d):  # control
            for y in range(d):  # target
                input_index = x * d + y
                output_y = (y - x) % d
                output_index = x * d + output_y
                U[output_index, input_index] = 1
        return U

    def _circuit_diagram_info_(self, args):
        return [f"C", f"ShiftDown(d={self.d})"]



class R_ij(cirq.Gate):
    """Generilzed rotation gate 

    """    

    def __init__(self, theta: float, d: int, i: int, j: int):
        """initialzier for rotation gate 

        Args:
            theta (float): angle of rotation
            d (int): dimension of rotated qudits 
            i (int): begining state
            j (int): state to be rotated into 
        """        
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





    




def full_U(k_i:int,i:int,d:int, anc_quds: list[cirq.LineQubit], psi_cops: list[list[cirq.LineQid]], ):
    

    num_op=h(i,d)


    for j in range(len(anc_quds[0])):
        gate= cirq.MatrixGate(expm(2j*np.pi * (num_op-k_i/len(psi_cops[0])*np.identity(d)) / (2**(j+1))),name=f"U_^{j+1}",qid_shape=(d,))

        for k in range(len(psi_cops[0])):
            yield cirq.ControlledGate(
                gate,
                control_values=(1,),
                control_qid_shape=(2,)
            ).on(anc_quds[i-1][j], psi_cops[j][k])  











def qpe(k_vec: tuple[int], num_qubs: int, d: int,n:int,  working_vec: list[cirq.Qid], mode=0,loc: int=0):


    anc_qubs=[]

    n=sum(k_vec)
  
    for i in range(d-1):

        temp=[]
        for j in range(num_qubs):
            temp.append(cirq.LineQubit(i*num_qubs+j))

        anc_qubs.append(temp)

 
    pos=loc
    
    master_quds=[] # 3d list of qubits. each entry in master_quds has l copies of the n working qubits
    
    cop_quds=[]
    for j in range(num_qubs-1):
        temp=[]
        for i in range(n):
            temp.append(cirq.LineQid(pos,dimension=d))
            pos+=1
        cop_quds.append(temp)

    master_quds.append(cop_quds)

    for i in range(d-2):
        cop_quds=[]
        for j in range(num_qubs):
            temp=[]
            for i in range(n):
                temp.append(cirq.LineQid(pos,dimension=d))
                pos+=1
            cop_quds.append(temp)
        master_quds.append(cop_quds)

    # Fan out operation 

    cshift=CShiftUpGate(d)

    for copy_quds in master_quds:
        for j in range(n):
            for cop in copy_quds:
                yield cshift(working_vec[j], cop[j])
    
    # Apply Hadamards to ancillas
    yield cirq.Moment([ cirq.H.on(qub) for qubs in anc_qubs for qub in qubs])
    
    print("before Unitaries")

    #Apply controlled-U^{x}

    master_quds[0].insert(0,working_vec)


    for i, k_i in enumerate(k_vec):
        if i==0:
            continue
        yield full_U(k_i,i,d,anc_qubs,master_quds[i-1])


    rcshift=CShiftDownGate(d)


    master_quds[0].pop(0)

    for copy_quds in reversed(master_quds):
        for j in reversed(range(n)):
            for cop in reversed(copy_quds):
                yield rcshift(working_vec[j], cop[j])


    print(f'after fan out')

    yield cirq.Moment([ cirq.H.on(qub) for qubs in anc_qubs for qub in qubs])


    print(' done circuit')
    if mode==0: # mode to measure or not 
        # Measure

        for i,qubs in enumerate(anc_qubs):
            yield cirq.measure(*qubs, key=f'k_{i+1}_estimate')
        
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
            yield R_ij(theta,d,i,i+1).on(qud)




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

    loc=l*(d-1)
    
    quds=[]
    for i in range(n):
        quds.append(cirq.LineQid(loc,dimension=d))
        loc+=1

    circuit=cirq.Circuit()


    circuit.append(init_qud(quds,k,n,d))

    circuit.append(qpe(k,l,d,n,quds,1,loc))

    # Simulate
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    print(circuit)
    print(cirq.dirac_notation(result.final_state_vector, qid_shape=(2,) * (l*(d-1)) + (d,)*n*l+(d,)*n*l*(d-2)))


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

    loc=l*(d-1)
    
    quds=[]
    for i in range(n):
        quds.append(cirq.LineQid(loc,dimension=d))
        loc+=1

    circuit=cirq.Circuit()


    circuit.append(init_qud(quds,k,n,d))

    circuit.append(qpe(k,l,d,n,quds,0,loc))

    # Simulate
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=shots)


    bitstrings = [["".join(str(bit) for bit in bits) for bits in result.measurements[f'k_{i+1}_estimate']] for i in range(d-1)]

    # Transpose to group bits by shot
    bitstrings_per_shot = list(zip(*bitstrings))  # Each element is a tuple of binary strings, one per ancilla group

    estimated_ks_all = [
        tuple(int(bits, 2) for bits in shot_bits)
        for shot_bits in bitstrings_per_shot
    ]

    # Count occurrences of each estimated k vector (excluding k0 calculation)
    k_vector_counter = Counter(estimated_ks_all)

    print(k_vector_counter,'\n')

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
    print(dickes)





k=(1,1,1)

dicke_shots(k,100)
#dicke_simulate(k)
