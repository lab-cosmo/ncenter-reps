from ase.io import read, write
import ase
import json

from copy import deepcopy
import numpy as np
import scipy as sp

from rascal.representations import SphericalExpansion, SphericalInvariants
from rascal.utils import (get_radial_basis_covariance, get_radial_basis_pca,
                          get_radial_basis_projections, get_optimal_radial_basis_hypers )
from rascal.utils import radial_basis
from rascal.utils import WignerDReal, ClebschGordanReal, spherical_expansion_reshape, lm_slice
ha2ev = 27.211386e3
from skcosmo.sample_selection import CUR, FPS

from ncnice import *

# reads structure and matrices
frame = read("data/qm7-example-structure.xyz", ":")[0]
fock = np.load("data/qm7-example-fock-pyscf.npy", allow_pickle=True)[0]
over = np.load("data/qm7-example-over-pyscf.npy", allow_pickle=True)[0]
orbs = json.load(open("data/qm7-example-orbs.json","r"))

# converts matrices in canonical form, and orthogonalizes the hamiltonian
fock = fix_pyscf_l1(fock, frame, orbs)
over = fix_pyscf_l1(over, frame, orbs)
ofock = lowdin_orthogonalize(fock, over)


#frame = read('data/ethanol-structures.xyz',':')[0]
#frame.cell=[100,100,100]
#frame.positions+=50
#orbs = json.load(open('data/ethanol-saph-orbs.json', "r"))
#ofock = np.load('data/ethanol-saph-ofock.npy', allow_pickle=True)[0]


# spherical expansion parameters
spherical_expansion_hypers = {
    "interaction_cutoff": 7,
    "max_radial": 6,
    "max_angular": 4,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.5,
    "radial_basis": "GTO",
    "global_species": [1,6,7,8],
    "cutoff_function_type": "RadialScaling",
    "cutoff_function_parameters": {"rate": 2, "scale": 1, "exponent": 2},
    "expansion_by_species_method": "user defined"
}

# start computing CG stuff, and convert targets in coupled form
mycg = ClebschGordanReal(spherical_expansion_hypers["max_angular"])

ofock_blocks = matrix_to_blocks(ofock, frame, orbs)
ofock_coupled = couple_blocks(ofock_blocks, mycg)

print("Computing features")
feats = compute_hamiltonian_representations([frame], orbs, spherical_expansion_hypers, 2, nu=1, cg=mycg, scale=1e3)

print("Fitting model")
FR = FockRegression(orbs, alpha=1e-8, fit_intercept="auto")
FR.fit(feats, ofock_coupled)

print("Predicting model")
pred_coupled = FR.predict(feats)
pred = blocks_to_matrix(decouple_blocks(pred_coupled, mycg), frame, orbs)

print("RMSE Hamiltonian", np.linalg.norm(ofock-pred)/np.sqrt(len(ofock)))
print("MAE Eigenvalues", np.mean(np.abs(np.linalg.eigvalsh(ofock)-np.linalg.eigvalsh(pred))))


# now we do transformed structures and check all is well

print("Checking rotations")
mywd = WignerDReal(spherical_expansion_hypers["max_angular"], *np.random.uniform(0,np.pi,size=3)) # random rotation
frame_rot = frame.copy()
mywd.rotate_frame(frame_rot)

feats_rot = compute_hamiltonian_representations([frame_rot], orbs, spherical_expansion_hypers, 2, nu=1, cg=mycg, scale=1e3)
pred_rot = blocks_to_matrix(decouple_blocks(FR.predict(feats_rot), mycg), frame, orbs)
print("RMSE Hamiltonian (pred)", np.linalg.norm(pred_rot-pred)/np.sqrt(len(ofock)))
print("MAE Eigenvalues (pred)", np.mean(np.abs(np.linalg.eigvalsh(pred_rot)-np.linalg.eigvalsh(pred))))

print("Checking inversion")
frame_inv = frame.copy()
frame_inv.positions = 100-frame.positions
feats_inv = compute_hamiltonian_representations([frame_inv], orbs, spherical_expansion_hypers, 2, nu=1, cg=mycg, scale=1e3)
pred_inv = blocks_to_matrix(decouple_blocks(FR.predict(feats_inv), mycg), frame, orbs)
print("RMSE Hamiltonian (pred)", np.linalg.norm(pred_inv-pred)/np.sqrt(len(ofock)))
print("MAE Eigenvalues (pred)", np.mean(np.abs(np.linalg.eigvalsh(pred_inv)-np.linalg.eigvalsh(pred))))

print("Checking atom permutations")
iperm = np.arange(len(frame.numbers), dtype=int)
np.random.shuffle(iperm)
frame_prm = frame.copy()
frame_prm.numbers = frame_prm.numbers[iperm]
frame_prm.positions = frame_prm.positions[iperm]
feats_prm = compute_hamiltonian_representations([frame_prm], orbs, spherical_expansion_hypers, 2, nu=1, cg=mycg, scale=1e3)
pred_prm = blocks_to_matrix(decouple_blocks(FR.predict(feats_prm), mycg), frame, orbs)
print("RMSE Hamiltonian (pred)", np.linalg.norm(pred_prm-pred)/np.sqrt(len(ofock)))
print("MAE Eigenvalues (pred)", np.mean(np.abs(np.linalg.eigvalsh(pred_prm)-np.linalg.eigvalsh(pred))))
