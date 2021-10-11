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
frame = read("data/qm7-example-structure.xyz", ":")
fock = np.load("data/qm7-example-fock-pyscf.npy", allow_pickle=True)
over = np.load("data/qm7-example-over-pyscf.npy", allow_pickle=True)
orbs = json.load(open("data/qm7-example-orbs.json","r"))

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




