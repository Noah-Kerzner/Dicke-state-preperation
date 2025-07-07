import cirq
import numpy as np
import itertools
from scipy.special import factorial







class XijGate(cirq.Gate):
    def __init__(self, d: int, i: int, j: int):
        assert 0 <= i < d and 0 <= j < d and i != j, "i and j must be distinct and in range"
        self.d = d
        self.i = i
        self.j = j

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        mat = np.eye(self.d, dtype=complex)
        mat[self.j, self.i] = 1  # Set transition i -> j
        mat[self.i, self.i] = 0  # Zero out original diagonal
        return mat

    def _circuit_diagram_info_(self, args):
        return f"X^{self.i}->{self.j}"







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







class a_sets:



    def __init__(self,k_vec,n):
        self.k_vec=k_vec
        self.d=len(k_vec)
        self.n=n
        self._j_forw, self._j_rev, self.chi = make_a_lists(k_vec,n )
        


    def j_forward(self,l:int,set: np.ndarray ) -> int:

        return self._j_forw[l][tuple(set)]
    
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

    def get_dit_thetas_2(self, i, p):
        thetas = []
        gamms=[]
        if p not in self._j_rev[i - 1]:
            return [0] * (self.d - 1)
        a = self.j_reverse(i - 1, p)

        for m in range(self.d - 1):
            gamma_val = self.gamma(i, p, m)

            if gamma_val == 0:
                a_new = a + m_hat(m, self.d)
                if tuple(a_new) in self._j_forw[i]:
                    thetas.append(0)
                else:
                    thetas.append(np.pi)
                continue

            # Compute denominator from previously calculated thetas
            denom = list_prod(thetas)
            if denom < 1e-10:
                thetas.append(0)
                continue

            val = gamma_val / denom
            val = np.clip(val, -1, 1)
            theta = 2 * np.arccos(val)
            thetas.append(theta)

        return thetas

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








def ditIgate(quds:cirq.LineQid, anc_qud: cirq.NamedQid, qub_anc: cirq.NamedQubit , sets: a_sets, i: int, a: np.ndarray,):
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

    
    p=sets.j_forward(i-1,a)
    thetas = sets.get_dit_thetas(i,p)

    d= sets.d
    chi=sets.chi

    # if wokring qudit is 0 sift anc dit up by one 
    yield cirq.ControlledGate(cirq.X, num_controls=2, control_values=(0,p,), control_qid_shape=(d,chi))(quds[i - 1], anc_qud,qub_anc)


    # 2s controlled rotations 
    # for j in range(d-1):
    #     yield cirq.ControlledGate(
    #         ditrotation(thetas[j], j, j + 1, d),
    #         num_controls=2,
    #         control_values=(1,p,),
    #         control_qid_shape=(2,chi,)
    #     )(qub_anc, anc_qud, quds[i - 1])

    for j in range(d-1):
        yield cirq.ControlledGate(
            ditrotation(thetas[j], j, j + 1, d),
        )(qub_anc, quds[i - 1])

    for j in range(d):
        a_new=a + m_hat(j,d)
        if tuple(a_new) in sets._j_forw[i]:
            p_new=sets.j_forward(i,a_new)
            if p==p_new:
                continue
            #print(p,p_new)
            yield cirq.ControlledGate(
                XijGate(chi,p,p_new),
                num_controls=2,
                control_values=(j,1,),
                control_qid_shape=(d,2,)
            )(quds[i-1],qub_anc, anc_qud)

    # Shift up by working qubit val - 1 
    for j in range(d):
        a_new=a + m_hat(j,d)
        if tuple(a_new) in sets._j_forw[i]:
            p_new=sets.j_forward(i,a_new)
            yield cirq.ControlledGate(cirq.X, num_controls=2, control_values=(j,p_new,), control_qid_shape=(d,chi))(quds[i - 1], anc_qud,qub_anc)












def ditUgate(quds, anc_qud,qub_anc, set: a_sets ,i:int):
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
    for a in list(cur.values()):
        yield ditIgate(quds, anc_qud,qub_anc,set, i, a)

   
    

            



def ditU(qubs, qud, qub_anc, set:a_sets):
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
        yield ditUgate(qubs, qud,qub_anc, set,i)










def qudit_dicke(k: tuple):


    sim = cirq.Simulator()

    n=int(sum(np.array(k)))


    
    set=a_sets(k,n)

    #print(set.gamma(1,0,1))
    #print(tuple(np.ndarray([0,0,1])+m_hat(2,3)) in set._j_forw[2])
    quds = cirq.LineQid.range(n, dimension=len(k))
    qudit = cirq.NamedQid('a', dimension=set.chi)
    qubit=cirq.NamedQubit('b')


    circuit = cirq.Circuit(ditU(quds, qudit,qubit, set))

    print("\n=== Final State Vector ===")
    result = sim.simulate(circuit)
    #print(circuit)
    print(f'dim = {set.chi}')
    print(cirq.dirac_notation(result.final_state_vector, qid_shape=(len(k),) * n + (set.chi,)+ (2,)))







if __name__=="__main__":
   
   

    k=(1,2,1)
    qudit_dicke(k)


    

    # x,y,z=make_a_lists(k,sum(k))
    # # for i in y:
    # #     print(i)
    # print(y[5])