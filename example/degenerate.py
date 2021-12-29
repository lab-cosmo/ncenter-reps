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
from ncnice import *

# initialize spherical expansion hypers and compute optimal basis
print("Init CGs")
spex_hypers = {
    "interaction_cutoff": 4,
    "max_radial": 4,
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

mycg = ClebschGordanReal(spex_hypers["max_angular"])

data = 'data/powerspectrum-triplet.xyz'
print("Computing optimal radial basis")
frames = read(data,':')
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

frames = read(data,':')
for f in frames:
    f.cell=[100,100,100]
    f.positions+=50

feats = {
    "rho_i^1;00" : [], 
    "rho_i^2;00" : [],
    "rho_i^3;00" : [], 
    "rho_ij^1;00" : [],
    "rho_ij^2;00" : [],
    "rho_ij^(0,1);00" : [],
    "rho_ij^(1,1);00" : [],
    "rho_i^(1<-1);00" : [],
}

for f in frames:
    rhoi = compute_rhoi(f, spex, spex_hypers)
    gij = compute_gij(f, spex_gij, hypers_ij)
    rho1i_l, prho1i_l = compute_rho1i_lambda(rhoi, 0, mycg)
    feats["rho_i^1;00"].append( rho1i_l[...,np.where(prho1i_l==1)[0],:])
    rho1ij = compute_rho1ij(rhoi, gij, mycg)
    rho2i_l, prho2i_l = compute_rho2i_lambda(rhoi, 0, mycg)
    feats["rho_i^2;00"].append(rho2i_l[...,np.where(prho2i_l==1)[0],:] )
    rho1ij_l, prho1ij_l = compute_rho1ij_lambda(rhoi, gij, 0, mycg)
    feats["rho_ij^1;00"].append(rho1ij_l[...,np.where(prho1ij_l==1)[0],:])
    rho2i_l_all, prho2i_l_all = compute_all_rho2i_lambda(rhoi, mycg)
    rho3i_l, prho3i_l = compute_rho3i_lambda(rho2i_l_all,rhoi, 0, mycg,prho2i_l_all)
    feats["rho_i^3;00"].append(rho3i_l[...,np.where(prho3i_l==1)[0],:] )
    rho2ij_l, prho2ij_l = compute_rho2ij_lambda(rho2i_l_all, gij, 0, mycg, prho2i_l_all)
    feats["rho_ij^2;00"].append(rho2ij_l[...,np.where(prho2ij_l==1)[0],:])
    rho1ijp_l, prho1ijp_l = compute_rho1ijp_lambda(rhoi, gij, 0, mycg)
    feats["rho_ij^(0,1);00"].append(rho1ijp_l[...,np.where(prho1ijp_l==1)[0],:])
    rho1ijp_l_all, prho1ijp_l_all = compute_all_rho1ijp_lambda(rhoi, gij, mycg)
    rho11ijp, prho11ijp = compute_rho11ijp_lambda(rhoi, rho1ijp_l_all, 0, mycg, prho1ijp_l_all)
    feats["rho_ij^(1,1);00"].append(rho11ijp[...,np.where(prho11ijp==1)[0],:])
    rho11P, prho11P = contract_rhoij(rho11ijp, prho11ijp, frames[0].symbols, ["H", "C", "O"])
    feats["rho_i^(1<-1);00"].append(rho11P[...,np.where(prho11P==1)[0],:])
    
for k in feats:
    print(f"Environment distances from frame 0, features type: {k}")
    for f in range(1,len(frames)):        
        print(np.sqrt(
            ((feats[k][0]-feats[k][f])**2).reshape(
               (len(feats[k][f]),-1)).sum(axis=1)/
            ((feats[k][0])**2).reshape(
               (len(feats[k][f]),-1)).sum(axis=1)   
               ))
    
