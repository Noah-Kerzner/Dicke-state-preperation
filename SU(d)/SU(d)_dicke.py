import cirq
import numpy as np
import itertools
from scipy.special import comb
from scipy.special import factorial




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
        if self.alpha == 0:
            return np.eye(dim, dtype=np.complex128)
        return np.roll(np.eye(dim, dtype=np.complex128), shift=self.alpha, axis=0)

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
        if self.alpha == 0:
            return np.eye(dim, dtype=np.complex128)
        return np.roll(np.eye(dim, dtype=np.complex128), shift=-self.alpha, axis=0)

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

    def get_sparse_dit_thetas_chain(self, i, p):
        # Step 1: find relevant levels
        active_levels = []
        gamma_values = []
        for m in range(self.d):
            g = self.gamma(i, p, m)
            if g != 0:
                active_levels.append(m)
                gamma_values.append(g)

        # Step 2: connect them with rotations
        thetas = []
        for idx in range(len(active_levels) - 1):
            a = active_levels[idx]
            b = active_levels[idx + 1]
            gamma_val = gamma_values[idx + 1]  # this is a choice — can also use avg or next
            denom = list_prod([theta for _,z,theta in thetas])
            if denom < 1e-10:
                theta = 0
            else:
                val = np.clip(gamma_val / denom, -1, 1)
                theta = 2 * np.arccos(val)
            thetas.append((a, b, theta))

        #print(f'gamms/thetas for i= {i}, p= {p}\n{gamma_values}\n{thetas}\n')
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


def ditIgate(quds:cirq.LineQid, anc_qud: cirq.NamedQid, sets: a_sets, i: int, p: int,):
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

    thetas = sets.get_dit_thetas(i,p)

    # if i==2 and p ==1:
    #     thetas[1]=np.pi
    # elif i==2 and p==2:
    #     thetas[1]==0
    d= sets.d
    chi=sets.chi
    # if wokring qudit is 0 sift anc dit up by one 
    yield cirq.ControlledGate(ditplus(chi-1, 1), num_controls=1, control_values=(0,), control_qid_shape=(d,))(quds[i - 1], anc_qud)

    # shift anc qudit down by x for working qudit value of x>=2
    for j in range(2, d):
        yield cirq.ControlledGate(ditminus(chi-1, j - 1), num_controls=1, control_values=(j,), control_qid_shape=(d,))(quds[i - 1], anc_qud)

    # 2s controlled rotations 
    for j in range(d-1):
        yield cirq.ControlledGate(
            ditrotation(thetas[j], j, j + 1, d),
            num_controls=1,
            control_values=((p + 1) % (chi),),
            control_qid_shape=(chi,)
        )(anc_qud, quds[i - 1])

    # Shift anc dit down one if working qudit is 0 
    yield cirq.ControlledGate(ditminus(chi-1, 1), num_controls=1, control_values=(0,), control_qid_shape=(d,))(quds[i - 1], anc_qud)

    # Shift up by working qubit val - 1 
    for j in range(2, d):
        yield cirq.ControlledGate(ditplus(chi-1, j - 1), num_controls=1, control_values=(j,), control_qid_shape=(d,))(quds[i - 1], anc_qud)





def sparse_ditIgate(quds: cirq.LineQid, anc_qud: cirq.NamedQid, sets: a_sets, i: int, p: int):
    d = sets.d
    chi = sets.chi

    theta_chain = sets.get_sparse_dit_thetas_chain(i, p)
    yield cirq.ControlledGate(ditplus(chi - 1, 1), num_controls=1, control_values=(0,), control_qid_shape=(d,))(quds[i - 1], anc_qud)

    for j in range(2, d):
        yield cirq.ControlledGate(ditminus(chi - 1, j - 1), num_controls=1, control_values=(j,), control_qid_shape=(d,))(quds[i - 1], anc_qud)


    if sets.gamma(i, p, 0) == 0:
        # Find the closest m > 0 with non-zero gamma
        target_level = None
        for m in range(1, d):
            if sets.gamma(i, p, m) != 0:
                target_level = m
                break
        if target_level is not None:
            yield cirq.ControlledGate(
                ditrotation(np.pi, 0, target_level, d),
                num_controls=1,
                control_values=((p + 1) % chi,),
                control_qid_shape=(chi,)
            )(anc_qud, quds[i - 1])

    for a,b, theta in theta_chain:
        yield cirq.ControlledGate(
            ditrotation(theta, a,b, d),
            num_controls=1,
            control_values=((p + 1) % chi,),
            control_qid_shape=(chi,)
        )(anc_qud, quds[i - 1])

    yield cirq.ControlledGate(ditminus(chi - 1, 1), num_controls=1, control_values=(0,), control_qid_shape=(d,))(quds[i - 1], anc_qud)

    for j in range(2, d):
        yield cirq.ControlledGate(ditplus(chi - 1, j - 1), num_controls=1, control_values=(j,), control_qid_shape=(d,))(quds[i - 1], anc_qud)





def ditUgate(quds, anc_qud, set: a_sets,i):
    """Indiviudal U of i. Code breaks U initary into simpler products of I gates for qudit extension

    Args:
        quds (cirq.LineQid): List of working qudits
        anc_qud (cirq.NamedQid): anciliry qudit
        n (int): number of working qudits
        k (int): number of spin operators
        i (int): the curent working qudit number
        s (float): the spin of the system

    """  


    for p in range(set.chi):
        #yield sparse_ditIgate(quds, anc_qud,set, i, p)
        yield ditIgate(quds, anc_qud,set, i, p)

        if i==2 and p==0:
            for k in range( set.d):
                yield cirq.ControlledGate(ditminus(set.chi - 1, 1), num_controls=1, control_values=(k,), control_qid_shape=(set.d,))(quds[i - 1], anc_qud)

   
    
 
            



def ditU(qubs, qud, set:a_sets):
    """
    Full unitary U to run U gates on each dit 

    Args:
        quds (cirq.LineQid): List of working qudits
        anc_qud (cirq.NamedQid): anciliry qudit
        n (int): number of working qudits
        k (int): number of spin operators
        s (float): the spin of the system

    """   
    count=1 
    for i in range(1,2):#range(1, set.n+ 1):
        yield ditUgate(qubs, qud, set,i)
        if i==2:
            yield cirq.ControlledGate(ditminus(set.chi-1, 1), num_controls=1, control_values=(0,), control_qid_shape=(set.d,))(qubs[i - 1], qud)

        #* This conditino is for (2,1,1)
        # if i>np.floor(set.n/2):
        #     yield ditminus(set.chi-1,count).on(qud)
        #     count+=1

    #* This set for (1,1,2)
    #     yield ditminus(set.chi-1,i-1).on(qud)
    # yield ditminus(set.chi-1,i-1).on(qud)

def qudit_dicke():
    sim = cirq.Simulator()

    k=(1,1,1,1)
    n=4

    
    set=a_sets(k,n)

    #print(set.gamma(1,0,1))
    #print(tuple(np.ndarray([0,0,1])+m_hat(2,3)) in set._j_forw[2])
    quds = cirq.LineQid.range(n, dimension=len(k))
    qudit = cirq.NamedQid('a', dimension=set.chi)



    circuit = cirq.Circuit(ditU(quds, qudit, set))

    print("\n=== Final State Vector ===")
    result = sim.simulate(circuit)
    #print(circuit)
    print(f'dim = {set.chi}')
    print(cirq.dirac_notation(result.final_state_vector, qid_shape=(len(k),) * 1 + (set.chi,)))







#qudit_dicke()

#print(multinomial(4,np.array([1,1,1,1])))

k=(1,1,1)
n=3
set=a_sets(k,n)
print(set.gamma(2,0,0))
print(set._j_rev)





# qud = cirq.LineQid.range(1, dimension=3)[0]  # get the single qudit

# sim = cirq.Simulator()
# circuit = cirq.Circuit()
# circuit.append(ditrotation(np.pi,0, 2, 3).on(qud))
# result = sim.simulate(circuit)
# print(circuit)
# print(cirq.dirac_notation(result.final_state_vector, qid_shape=(3,)))

#print(general_U_matrix(3)[2])

