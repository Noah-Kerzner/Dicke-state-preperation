# qudit Dicke state preparation
A collection of Jupyter notebooks and .py files to simulate the preparation of qudit Dicke states using `CIRQ`

Based on "Simple ways of preparing qudit Dicke states", by Noah B. Kerzner, Federico Galeazzi and Rafael I. Nepomechie

https://arxiv.org/abs/2507.13308

Can run on Google Colab  (  https://colab.research.google.com ) or on a local installation of python and jupyter 

## Files

- `line-supp.py` : Proof of concept for 2 linear superpostions of $SU(2)$ spin - $s$ Dicke States by $O(1)$ QPE through Hadamard test

### .ipynb files in `spin-s` folder

- `SU(2)_spin-s_mps.ipynb` : $SU(2)$ spin - $s$ Dicke state preparation based on MPS, __with examples__ 
- `SU(2)_spin-s_qpe_O(1).ipynb` : $SU(2)$ spin - $s$ Dicke state preparation based on QPE in $O(1)$ depth, __with examples__
- `SU(2)_spin-s_qpe_O(log(n)).ipynb` : $SU(2)$ spin - $s$ Dicke state preparation based on QPE in $O(log(sn))$ depth, __with examples__
- `SU(2)_spin-s_qpe_hadamard.ipynb` : $SU(2)$ spin - $s$ Dicke state preparation based on QPE - Hadamard test, __with examples__

### .py files in `spin-s` folder

- `SU(2)_spin-s_mps.py`: $SU(2)$ spin - $s$ Dicke state preparation based on MPS
- `SU(2)_spin-s_qpe_O(1).py` : $SU(2)$ spin - $s$ Dicke state preparation based on QPE in $O(1)$ depth
- `SU(2)_spin-s_qpe_O(log(n)).py` : $SU(2)$ spin - $s$ Dicke state preparation based on QPE in $O(log(sn))$ depth
- `SU(2)_spin-s_qpe_hadamard.py` : $SU(2)$ spin - $s$ Dicke state preparation based on QPE - Hadamard test

### .ipynb files in `SU(d)` folder

- `SU(d)_mps.ipynb` : $SU(d)$ Dicke state preparation based on MPS, __with examples__
- `SU(d)_qpe_O(1).ipynb` : $SU(d)$ Dicke state preparation based on QPE in $O(1)$ depth, __with examples__ -- this code requires extensive memory due to amount of ancillas
- `SU(d)_qpe_O(log(n)).ipynb` : $SU(d)$ Dicke state preparation based on QPE in $O(log(n))$ depth, __with examples__
- `SU(d)_qpe_hadamard.ipynb` : $SU(d)$ Dicke state preparation based on QPE - Hadamard test, __with examples__

### .py files in `SU(d)` folder

- `SU(d)_mps.py`: $SU(d)$ Dicke state preparation based on MPS
- `SU(d)_qpe_O(1).py` : $SU(d)$ Dicke state preparation based on QPE in $O(1)$ depth -- this code requires extensive memory due to amount of ancillas
- `SU(d)_qpe_O(log(n)).py` : $SU(d)$ Dicke state preparation based on QPE in $O(log(n))$ depth
- `SU(d)_qpe_hadamard.py` : $SU(d)$ Dicke state preparation based on QPE - Hadamard test

## Notes 

- As stated above the $SU(d)$ qpe in $O(1)$ is difficult to simulate because of ancilla count. **Do not use for systems with more then 20 ancillas**
- Operation of each file described by comments in file
