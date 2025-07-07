# Dicke-state-preperation
A collection of Jupyter notebooks and .py files to simulate the preparation of qudit Dicke states using `CIRQ`


## Files

- `line-supp.py` : Proof of concept for 2 linear superpostions of $SU(2)$ spin-$s$ Dicke States by $O(1)$ qpe through Hadamard test

### .py files in `spin-s` folder

- `correct_qudit_qpe.py` : $SU(2)$ spin-$s$ qpe simulator in $O(log(2sn))$ depth
- `general_dicke_state.py`: $SU(2)$ spin-$s$ unitary preperation simulator
- `O(1)_spin-s_qpe.py` : $SU(2)$ spin-$s$ qpe simulator in $O(1)$ depth

### .py files in `SU(d)` folder

- `SU(d)_qpe.py` : $SU(d)$ qpe simulator in $O(dlog(n))$ depth
- `SU(d)_mps.py`: $SU(d)$ unitary preperation by mps simulator
- `O(1)_SU(d)_qpe.py` : $SU(d)$ qpe in $O(1)$ depth -- this code functions but requires extensive memeory due to amount of ancillas 


### .ipynb files in `notebooks` folder

- `SU(2)_spin-s_mps.ipynb` : $SU(2)$ spin-$s$ unitary preperation simulator with examples
- `SU(2)_spin-s_O(1)_qpe.ipynb` : $SU(2)$ spin-$s$ qpe simulator in $O(1)$ depth with examples
- `SU(2)_spin-s_qpe_O(log(n)).ipynb` : $SU(2)$ spin-$s$ qpe simulator in $O(log(2sn))$ depth with examples
- `SU(d)_dicke_states.ipynb` : $SU(d)$ $SU(d)$ unitary preperation simulator with examples
- `SU(d)_O(1)_qpe.ipynb` : $SU(d)$ qpe in $O(1)$ depth with examples -- this code functions but requires extensive memeory due to amount of ancillas
- `SU(d)_O(log(n))_qpe.ipynb` : $SU(d)$ qpe simulator in $O(dlog(n))$ depth with examples

## Notes 

- As stated above the $SU(d)$ qpe in $O(1)$ is difficult to simulate because of ancilla count. **Do not use for systems with more then 20 ancillas**
- Operation of each file described by comments in file
