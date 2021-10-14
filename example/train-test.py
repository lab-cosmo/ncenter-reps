from ase.io import read, write
import ase
import json

from tqdm import tqdm
class tqdm_reusable:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __iter__(self):
        return tqdm(*self._args, **self._kwargs).__iter__()

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

# initialize spherical expansion hypers and compute optimal basis
print("Init CGs")
spex_hypers = {
    "interaction_cutoff": 5,
    "max_radial": 5,
    "max_angular": 3,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.5,
    "radial_basis": "GTO",
    "global_species": [1,6,8],
    "cutoff_function_type": "RadialScaling",
    "cutoff_function_parameters": {"rate": 2, "scale": 1, "exponent": 2},
    "expansion_by_species_method": "user defined"
}

spex = SphericalExpansion(**spex_hypers)
mycg = ClebschGordanReal(spex_hypers["max_angular"])

print("Computing optimal radial basis")
nframes = 1000
frames = read('data/ethanol-structures.xyz',':')[:nframes]
for f in frames:
    f.cell=[100,100,100]
    f.positions+=50
    # use same specie for all atoms so we get a single projection matrix,
    # which we can apply throughout. a bit less efficient but much more practical
    f.numbers = f.numbers*0+1
spex_hypers = get_optimal_radial_basis_hypers(spex_hypers, frames, expanded_max_radial=16)
# ... and we replicate it
pm = spex_hypers['optimization']['RadialDimReduction']['projection_matrices'][1]
spex_hypers['optimization']['RadialDimReduction']['projection_matrices'] = { i: pm for i in range(99) }
spex = SphericalExpansion(**spex_hypers)
hypers_ij = deepcopy(spex_hypers)
hypers_ij["expansion_by_species_method"] = "structure wise"
spex_gij = SphericalExpansion(**hypers_ij)


orbs = json.load(open('data/ethanol-saph-orbs.json', "r"))
frames = read('data/ethanol-structures.xyz', ':')[:nframes]
ofocks = np.load('data/ethanol-saph-ofock.npy', allow_pickle=True)[:nframes]

# hamiltonian to block coupling
ofock_blocks, slices_idx = matrix_list_to_blocks(ofocks, frames, orbs, mycg)

# training settings
train_fraction = 0.5
itrain = np.arange(len(frames))
np.random.shuffle(itrain)
ntrain = int(len(itrain)*train_fraction)
itest = itrain[ntrain:]; itrain=itrain[:ntrain]
train_slices = get_block_idx(itrain, slices_idx)

FR = FockRegression(orbs, alphas=np.geomspace(1e-8, 1e4, 7),
                    fit_intercept="auto")

for f in frames:
    f.cell=[100,100,100]
    f.positions+=50

print("Calling all representation subroutines (no PCA)")
rhoi = compute_rhoi(frames[0], spex, spex_hypers)
gij = compute_gij(frames[0], spex_gij, hypers_ij)
rho1ij = compute_rho1ij(rhoi, gij, mycg)
rho1i_l = compute_rho1i_lambda(rhoi, spex_hypers["max_angular"], mycg)
rho2i_l, prho2i_l = compute_rho2i_lambda(rhoi, spex_hypers["max_angular"], mycg)
rho2i_l_all, prho2i_l_all = compute_all_rho2i_lambda(rhoi, mycg)
rho0ij_l = compute_rho0ij_lambda(rhoi, gij, spex_hypers["max_angular"], mycg)
rho1ij_l = compute_rho1ij_lambda(rhoi, gij, spex_hypers["max_angular"], mycg)
rho2ij_l = compute_rho2ij_lambda(rho2i_l_all, gij, spex_hypers["max_angular"], mycg, prho2i_l_all)

print("Regression model (nu=0 no PCA)")
feats = compute_hamiltonian_representations(tqdm_reusable(frames, desc="features", leave=False),
                        orbs, spex_hypers, 2, nu=0, cg=mycg, scale=1e3)

FR.fit(feats, ofock_blocks, train_slices, progress=tqdm)
pred_blocks = FR.predict(feats, progress=tqdm)
pred_ofocks = blocks_to_matrix_list(pred_blocks, frames, slices_idx, orbs, mycg)

mse_train = 0
for i in itrain:
    mse_train += np.sum((pred_ofocks[i] - ofocks[i])**2)/len(ofocks[i])/len(itrain)

mse_test = 0
for i in itest:
    mse_test += np.sum((pred_ofocks[i] - ofocks[i])**2)/len(ofocks[i])/len(itest)

print("Train RMSE: ", np.sqrt(mse_train))
print("Test RMSE: ", np.sqrt(mse_test))

print("Regression model (nu=1 no PCA)")
feats = compute_hamiltonian_representations(tqdm_reusable(frames, desc="features", leave=False),
                        orbs, spex_hypers, 2, nu=1, cg=mycg, scale=1e3)

FR.fit(feats, ofock_blocks, train_slices, progress=tqdm)
pred_blocks = FR.predict(feats, progress=tqdm)
pred_ofocks = blocks_to_matrix_list(pred_blocks, frames, slices_idx, orbs, mycg)

mse_train = 0
for i in itrain:
    mse_train += np.sum((pred_ofocks[i] - ofocks[i])**2)/len(ofocks[i])/len(itrain)

mse_test = 0
for i in itest:
    mse_test += np.sum((pred_ofocks[i] - ofocks[i])**2)/len(ofocks[i])/len(itest)

print("Train RMSE: ", np.sqrt(mse_train))
print("Test RMSE: ", np.sqrt(mse_test))

print("Computing PCA")
rhoi = compute_rhoi(frames, spex, spex_hypers)
rhoi_pca, rhoi_pca_eva = compute_rhoi_pca(rhoi, npca=10)
print("rho_i singular values", rhoi_pca_eva[0]/rhoi_pca_eva[0][0])
crhoi = apply_rhoi_pca(rhoi, rhoi_pca)

rho2i_pca, rho2i_pca_eva = compute_rho2i_pca(crhoi, mycg, npca=80)
print("rho2_i singular values", rho2i_pca_eva[(0,1)]/rho2i_pca_eva[(0,1)][0])
rho2i_l, prho2i_l = compute_rho2i_lambda(crhoi, spex_hypers["max_angular"], mycg)
crho2i = apply_rho2i_pca(rho2i_l, prho2i_l, rho2i_pca)

rho1ij_pca, rho1ij_pca_eva = compute_rhoij_pca(frames, spex_hypers, mycg, nu=1, npca=80,
                    rho1i_pca = rhoi_pca)
print("rho1_ij singular values", rho1ij_pca_eva[(0,1)]/rho1ij_pca_eva[(0,1)][0])

rho2ij_pca, rho2ij_pca_eva = compute_rhoij_pca(frames, spex_hypers, mycg, nu=2, npca=80,
                    rho1i_pca = rhoi_pca, rho2i_pca = rho2i_pca)
print("rho2_ij singular values", rho2ij_pca_eva[(0,1)]/rho2ij_pca_eva[(0,1)][0])

print("Regression model (nu=2, with PCA)")
feats = compute_hamiltonian_representations(tqdm_reusable(frames, desc="features", leave=False),
                        orbs, spex_hypers, 2, nu=2, cg=mycg, scale=1e3,
                        rhoi_pca = rhoi_pca, rho2i_pca = rho2i_pca,
                        rhoij_pca = rho2ij_pca
                        )

FR.fit(feats, ofock_blocks, train_slices, progress=tqdm)
pred_blocks = FR.predict(feats, progress=tqdm)
pred_ofocks = blocks_to_matrix_list(pred_blocks, frames, slices_idx, orbs, mycg)

mse_train = 0
for i in itrain:
    mse_train += np.sum((pred_ofocks[i] - ofocks[i])**2)/len(ofocks[i])/len(itrain)

mse_test = 0
for i in itest:
    mse_test += np.sum((pred_ofocks[i] - ofocks[i])**2)/len(ofocks[i])/len(itest)

print("Train RMSE: ", np.sqrt(mse_train))
print("Test RMSE: ", np.sqrt(mse_test))
