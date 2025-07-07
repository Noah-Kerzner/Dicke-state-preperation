import cirq
import numpy as np
import itertools
from scipy.special import factorial




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




class FunctionControlledGate(cirq.Gate):
    def __init__(self, fxz_dict, d_control1, d_target, d_control2):
        """
        Implements a gate that maps |x>|y>|z> to |x>|f(x,z)>|z>, where f(x,z) is given by fxz_dict.

        Args:
            fxz_dict (dict): mapping from (x, z) tuple to target basis state f(x, z)
            d_control1 (int): dimension of first control qudit (x)
            d_target (int): dimension of target qudit (y)
            d_control2 (int): dimension of second control qudit (z)
        """
        self.fxz_dict = fxz_dict
        self.d_control1 = d_control1
        self.d_target = d_target
        self.d_control2 = d_control2

    def _num_qubits_(self):
        return 3

    def _qid_shape_(self):
        return (self.d_control1, self.d_target, self.d_control2)

    def _unitary_(self):
        dim = self.d_control1 * self.d_target * self.d_control2
        U = np.zeros((dim, dim), dtype=complex)
        for x in range(self.d_control1):
            for z in range(self.d_control2):
                f_xz = self.fxz_dict.get((x, z), None)
                for y in range(self.d_target):
                    input_index = x * self.d_target * self.d_control2 + y * self.d_control2 + z
                    if f_xz is None or not (0 <= f_xz < self.d_target):
                        # Identity action when mapping is not defined
                        output_index = input_index
                    else:
                        output_index = x * self.d_target * self.d_control2 + f_xz * self.d_control2 + z
                    U[output_index, input_index] = 1
        return U

    def _circuit_diagram_info_(self, args):
        return [f"x(d={self.d_control1})", f"f(x,z)", f"z(d={self.d_control2})"]



class ResetYGate(cirq.Gate):
    def __init__(self, reverse_mapping, d_x, d_y, d_z):
        self.reverse_mapping = reverse_mapping
        self.d_x = d_x  # j
        self.d_y = d_y  # y (to be reset)
        self.d_z = d_z  # p

    def _num_qubits_(self):
        return 3

    def _qid_shape_(self):
        return (self.d_x, self.d_y, self.d_z)

    def _unitary_(self):
        dim = self.d_x * self.d_y * self.d_z
        U = np.eye(dim, dtype=complex)
        for j in range(self.d_x):
            for p in range(self.d_z):
                if (j, p) not in self.reverse_mapping:
                    continue
                z = self.reverse_mapping[(j, p)]
                if z == 0:
                    continue  # no change needed
                # Swap |j>|z>|p> with |j>|0>|p>
                idx_z = j * self.d_y * self.d_z + z * self.d_z + p
                idx_0 = j * self.d_y * self.d_z + 0 * self.d_z + p
                # Swap rows
                U[[idx_z, idx_0], :] = U[[idx_0, idx_z], :]
        return U

    def _circuit_diagram_info_(self, args):
        return ["j", "reset_y", "p"]




class a_sets:



    def __init__(self,k_vec,n):
        self.k_vec=k_vec
        self.d=len(k_vec)
        self.n=n
        self._j_forw, self._j_rev, self.chi = make_a_lists(k_vec,n )
        


    def j_forward(self,l:int,vec: np.ndarray ) -> int:

        return self._j_forw[l][tuple(vec)]
    
    def j_reverse(self,l:int,label: int) -> np.ndarray:

        return self._j_rev[l][label]
    
    def gamma(self, i, p, m):
        if p in self._j_rev[i - 1]:
            a = self.j_reverse(i - 1, p)
            hat_m = m_hat(m, self.d)
            a_new = a + hat_m
            if tuple(a_new) in self._j_forw[i]:
                # If we've already used symbol m to its maximum allowed count, skip
                gamma_val = self._c(self.n, self.k_vec, i, a_new) * self._c(i, a_new, i - 1, a) / self._c(self.n, self.k_vec, i - 1, a)
                return gamma_val
        return 0
        

    def _c(self,n:int,k: np.ndarray,l: int, a:  np.ndarray)-> float:
    
        val = multinomial(l,a) * multinomial(n - l, k - a) / multinomial(n, k)
        return np.sqrt(val)


    def get_dit_thetas(self, i, p):
        thetas = []

        gamms=[]
        for m in range(self.d - 1):
            gamma_val = self.gamma(i, p, m)

            gamms.append(gamma_val)
            if gamma_val == 0:
                thetas.append(np.pi)
                continue

            denom = list_prod(thetas)
            
            if denom < 1e-10:
                thetas.append(0)
                continue

            val = gamma_val / denom
            val = np.clip(val, -1, 1)
            theta = 2 * np.arccos(val)

            thetas.append(theta)

        #print(f'gamms/thetas for i= {i}, p= {p}\n{gamms}\n{thetas}\n')
        return thetas

   



def multinomial(n: int, k: np.ndarray) -> float:
    """
    Computes the multinomial coefficient:
    multinomial(n; k_1, k_2, ..., k_r) = n! / (k_1! * k_2! * ... * k_r!)

    Args:
        n (int): total number of elements (sum of k)
        k (np.ndarray): array of counts in each category

    Returns:
        float: multinomial coefficient
    """
    return factorial(n) / np.prod(factorial(k))


    


def generate_a_l_k(k_vec, l):
    """
    Generates all vectors a = (a_0, ..., a_{d-1}) such that:
    - 0 <= a_i <= k_i for each i
    - sum(a) == l
    """
    ranges = [range(k_i + 1) for k_i in k_vec]
    all_candidates = itertools.product(*ranges)
    j_forward={}
    j_reverse={}
    label=0
    for a in reversed(list(all_candidates)):
        if sum(a)==l:
            valid_vec=np.array(a)
            j_forward[tuple(valid_vec)]=label
            j_reverse[label]=valid_vec
            label+=1
        
    return j_forward, j_reverse


def make_a_lists(k_vec,n):

    forward_j_list=[]
    reverse_j_list=[]

    max_len=0
    for i in range(n+1):
        j_for,j_rev=generate_a_l_k(k_vec,i)
        forward_j_list.append(j_for)
        reverse_j_list.append(j_rev)
        if i ==np.floor(n/2):
            max_len=len(list(j_for.keys()))

    
    return forward_j_list, reverse_j_list, max_len


def m_hat(m:int,d):
    vec=np.array([0]*d)
    vec[m]=1
    return vec




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








def ditUgate(quds, anc_qud,copy_qud, set: a_sets ,i:int):
    """Indiviudal U of i. Code breaks U initary into simpler products of I gates for qudit extension

    Args:
        quds (cirq.LineQid): List of working qudits
        anc_qud (cirq.NamedQid): anciliry qudit
        n (int): number of working qudits
        k (int): number of spin operators
        i (int): the curent working qudit number
        s (float): the spin of the system

    """  

    cur=set._j_rev[i-1]


    d=set.d
    chi=set.chi

    thetas=[]

    yield CShiftUpGate(chi,chi).on(anc_qud,copy_qud)
    
    

    for p in range(chi):
        thetas.append(set.get_dit_thetas(i,p))

    thetas=np.array(thetas)
    print(thetas)
    for j in range(int(d)-1):
        yield ControlledDitRotation(thetas[:,j],j,j+1,chi,d).on(anc_qud,quds[i-1])

    mappings = {}
    reverse_mapping = {}

    for a in list(cur.values()):
        z = set.j_forward(i - 1, a)
        for j in range(d):
            a_new = a + m_hat(j, d)
            if tuple(a_new) in set._j_forw[i]:
                p_new = set.j_forward(i, a_new)
                # mappings: (j, z) -> p_new means f(x=j, y=z) = p_new
                mappings[(j, z)] = p_new
                # reverse_mapping: (j, p_new) -> z means f⁻¹(x=j, z=p_new) = y=z
                reverse_mapping[(j, p_new)] = z


    
    print('\n',mappings,f'for i={i}')
   
    yield FunctionControlledGate(mappings,d,chi,chi).on(quds[i-1],anc_qud,copy_qud)
    yield ResetYGate(reverse_mapping,d,chi,chi).on(quds[i-1],copy_qud,anc_qud) # reset copy qud 

            



def ditU(qubs, qud,copy_qud, set:a_sets):
    """
    Full unitary U to run U gates on each dit 

    Args:
        quds (cirq.LineQid): List of working qudits
        anc_qud (cirq.NamedQid): anciliry qudit
        n (int): number of working qudits
        k (int): number of spin operators
        s (float): the spin of the system

    """   
    for i in range(1, set.n+ 1):
        yield ditUgate(qubs, qud,copy_qud, set,i)









def qudit_dicke(k: tuple):


    sim = cirq.Simulator()

    n=int(sum(np.array(k)))


    
    set=a_sets(k,n)


    quds = cirq.LineQid.range(n, dimension=len(k))
    qudit = cirq.NamedQid('a', dimension=set.chi)
    cop_qud= cirq.NamedQid('b', dimension=set.chi)


    circuit = cirq.Circuit(ditU(quds, qudit,cop_qud, set))

    print("\n=== Final State Vector ===")
    result = sim.simulate(circuit)
    print(circuit)
    print(f'dim = {set.chi}')
    print(cirq.dirac_notation(result.final_state_vector, qid_shape=(len(k),) * n + (set.chi,)*2))






if __name__=="__main__":
   
   

    k=(1,2,1)
    qudit_dicke(k)