from ase.io import read, write
import ase
import json

from copy import deepcopy
import numpy as np
import scipy as sp

from ncnice import *

# reads structure and matrices
frame = read("data/qm7-example-structure.xyz", ":")
fock = np.load("data/qm7-example-fock-pyscf.npy", allow_pickle=True)
over = np.load("data/qm7-example-over-pyscf.npy", allow_pickle=True)
orbs = json.load(open("data/qm7-example-orbs.json","r"))

ofock = deepcopy(fock)
# converts matrices in canonical form, and orthogonalizes the Hamiltonian
for i in range(len(frame)):
    fock[i] = fix_pyscf_l1(fock[i], frame[i], orbs)
    over[i] = fix_pyscf_l1(over[i], frame[i], orbs)
    ofock[i] = lowdin_orthogonalize(fock[i], over[i])

# these are the indices of "big basis" functions to be picked from the orbs listing
# to project the active space of the Hamiltonian
sel_types = {'C' : [1,3,4,5], 'O': [1,3,4,5], 'N': [1,3,4,5], 'H' : [0]}
# this is the number of "core states" to be discarded from the molecular orbitals for each species
n_core = {'C' : 1, 'O': 1, 'N': 1, 'H' : 0}

for i in range(len(frame)):
    print("Computing SAPH for ", frame[i].symbols)
    saph = compute_saph(fock[i], over[i], frame[i], orbs, sel_types, n_core)
    print("Full eigenvalues")
    print(sp.linalg.eigvalsh(fock[i], over[i]))
    print("SAPH eigenvalues")
    print(sp.linalg.eigvalsh(saph))
