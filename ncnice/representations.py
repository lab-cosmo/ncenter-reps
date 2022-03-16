import numpy as np
import scipy as sp
from copy import deepcopy
import itertools
from rascal.utils import spherical_expansion_reshape, lm_slice
from rascal.representations import SphericalExpansion

def compute_rhoi(frame, spex, hypers):
    # computes <nlm|rho_i> - the basic atom-centered equivariant
    return spherical_expansion_reshape(spex.transform(frame).get_features(spex), **hypers)

def compute_gij(frame, spex, hypers):
    # computes <nlm|rho_ij^0; g> - the basic pair equivariant
    # can be done all in one go by tagging each atom as being a separate species
    sel = frame.copy()
    sel.numbers[:]= range(len(frame.numbers))
    return spherical_expansion_reshape(spex.transform(sel).get_features(spex), **hypers)

def compute_rho1ij(rhoi, gij, cg):
    """ computes |rho^1_ij> - the pair invariant"""
    # natom, natom, nel, nmax, nmax, lmax+1
    shape = (rhoi.shape[0],  rhoi.shape[0],
             rhoi.shape[1],  rhoi.shape[2],
             gij.shape[2],  int(np.sqrt(rhoi.shape[3])))
    rho1ij = np.zeros(shape)

    for l in range(shape[5]):
        rho1ij[...,l] = cg.combine_einsum(rhoi[...,lm_slice(l)], gij[...,lm_slice(l)],
                                                L=0, combination_string="ian,ijN->ijanN")[...,0]
        return rho1ij

def compute_rho1i_lambda(rhoi, L, cg):
    """ computes |rho^1_i; lm>, one-center, radial equivariant (just a slice of the <nlm|rhoi>"""
    # natom, nel, nmax, lmax+1
    rhoi = rhoi.reshape((rhoi.shape[0], -1, rhoi.shape[-1]))
    parity = np.ones(rhoi.shape[1]) #spherical harmonics like components have parity 1 by our defintion since they are polar tensors
    return rhoi[..., lm_slice(L)], parity

def compute_rho2i_lambda(rhoi, L, cg):
    """ computes |rho^2_i; lm> - lambda-SOAP - using CG utilities from librascal"""

    lmax = int(np.sqrt(rhoi.shape[-1])) -1
    # can't work out analytically how many terms we have, so we precompute it here
    nl = 0
    for l1 in range(lmax + 1):
        for l2 in range(l1, lmax + 1):  # only need l2>=l1
            if l2 - l1 > L or l2 + l1 < L:
                continue
            nl += 1

    # natom, nel, nmax, nel, nmax, nl, M
    shape = (rhoi.shape[0],
             rhoi.shape[1],  rhoi.shape[2],
             rhoi.shape[1],  rhoi.shape[2],
             nl, 2*L+1)
    rho2ilambda = np.zeros(shape)
    parity = np.ones(nl, dtype = int)*(1-2*(L%2))

    il = 0
    for l1 in range(lmax+1):
        for l2 in range(l1, lmax+1):
            if l2 - l1 > L or l2 + l1 < L:
                continue
            rho2ilambda[...,il,:] = cg.combine_einsum(rhoi[...,lm_slice(l1)], rhoi[...,lm_slice(l2)],
                                                L, combination_string="ian,iAN->ianAN")
            parity[il] *= (1-2*(l1%2)) * (1-2*(l2%2))
            il += 1

    return rho2ilambda, parity


def compute_rho3i_lambda(rho2i_l, rhoi, L, cg, prho2i_l):
    """ computes |rho^3_i; lm> - lambda-SOAP - using CG utilities from librascal, takes a dictionary of
    rho2i[l] as input"""
    lmax = int(np.sqrt(rhoi.shape[-1])) -1
    index_l1l2={k:[] for k in range(lmax+1)} #to keep track of which (l1,l2) pair for given k
    # can't work out analytically how many terms we have, so we precompute it here
    nl = 0
    for l1 in range(0,lmax + 1):
        for l2 in range(l1, lmax + 1): # only need l2>=l1
            nl1l2=0
            for k in range(abs(l1-l2), min((l1+l2), lmax)+1): #intermediate coupling
                nl1l2+=1
                index_l1l2[k].append((l1,l2))
                for l3 in range(l2, lmax + 1): # only need l3>=l2
                    if abs(k - l3) > L or l3 + k < L:
                        continue
                    nl += 1

    shape = (rhoi.shape[0],
             rhoi.shape[1],  rhoi.shape[2],
             rhoi.shape[1],  rhoi.shape[2],
             rhoi.shape[1],  rhoi.shape[2],
             nl, 2*L+1)
    rho3ilambda = np.zeros(shape)
    parity = np.ones(nl, dtype = int)*(1-2*(L%2))

    il = 0
    for l1 in range(0, lmax + 1):
        for l2 in range(l1, lmax + 1): # only need l2>=l1
            for k in range(abs(l1-l2), min((l1+l2), lmax)+1): #intermediate coupling
                for l3 in range(l2, lmax + 1): # only need l3>=l2
                    if abs(l3 - k) > L or l3 + k < L:

                        continue
                    il1l2=  index_l1l2[k].index((l1,l2))
                    #print(l1, l2,k, il1l2)
                    rho3ilambda[...,il,:] = cg.combine_einsum(rhoi[..., lm_slice(l3)],
                                                              rho2i_l[k][...,il1l2,:],
                                                        L, combination_string="ian,iANbM->ianANbM")

                    parity[il] *= prho2i_l[k][il1l2]*(1-2*(l3%2))*(1-2*(k%2))

                    #print('l1, l2, k, l3, prho2, p, expected', l1, l2, k,l3,prho2i_l[k][il1l2], parity[il], (-1)**(l1+l2+l3+L))
                    il += 1
                il1l2+=1

    return rho3ilambda, parity


def compute_rho0ij_lambda(rhoi, gij, L, cg,  prfeats = None): # prfeats is (in analogy with rho2ijlambda) the parity, but is not really necessary)
    """ computes |rho^0_{ij}; lm> """
    rho0ij = gij[..., lm_slice(L)].reshape((gij.shape[0], gij.shape[1], -1, 2*L+1))
    return rho0ij, np.ones(rho0ij.shape[2])

def compute_rho1ij_lambda(rhoi, gij, L, cg, prfeats = None): # prfeats is (in analogy with rho2ijlambda) the parity, but is not really necessary
    """ computes |rho^1_{ij}; lm> """

    lmax = int(np.sqrt(gij.shape[-1])) -1
    # can't work out analytically how many terms we have, so we precompute it here
    nl = 0
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):  # |rho_i> and |rho_ij; g> are not symmetric so we need all l2
            if abs(l2 - l1) > L or l2 + l1 < L:
                continue
            nl += 1

    rhoi = rhoi.reshape((rhoi.shape[0], -1, rhoi.shape[-1]))
    # natom, natom, nel*nmax, nmax, lmax+1, lmax+1, M
    shape = (rhoi.shape[0], rhoi.shape[0],
             rhoi.shape[1] , gij.shape[2], nl, 2*L+1)
    rho1ijlambda = np.zeros(shape)
    parity = np.ones(nl, dtype = int)*(1-2*(L%2))

    il = 0
    for l1 in range(lmax+1):
        for l2 in range(lmax+1):
            if abs(l2 - l1) > L or l2 + l1 < L:
                continue
            rho1ijlambda[:,:,:,:,il] = cg.combine_einsum(rhoi[...,lm_slice(l1)], gij[...,lm_slice(l2)],
                                                L, combination_string="in,ijN->ijnN")
            parity[il] *= (1-2*(l1%2)) * (1-2*(l2%2))
            il+=1

    return rho1ijlambda, parity

def compute_rho1ijp_lambda(rhoi, gij, L, cg, prfeats = None): # prfeats is (in analogy with rho2ijlambda) the parity, but is not really necessary
    """ computes |rho^1p_{ij}; lm>, i.e. the pair term decorated with the description of the j-atom! """


    lmax = int(np.sqrt(gij.shape[-1])) -1
    # can't work out analytically how many terms we have, so we precompute it here
    nl = 0
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):  # |rho_i> and |rho_ij; g> are not symmetric so we need all l2
            if abs(l2 - l1) > L or l2 + l1 < L:
                continue
            nl += 1

    rhoi = rhoi.reshape((rhoi.shape[0], -1, rhoi.shape[-1]))
    # natom, natom, nel*nmax, nmax, lmax+1, lmax+1, M
    shape = (rhoi.shape[0], rhoi.shape[0],
             rhoi.shape[1] , gij.shape[2], nl, 2*L+1)
    rho1ijplambda = np.zeros(shape)
    parity = np.ones(nl, dtype = int)*(1-2*(L%2))

    il = 0
    for l1 in range(lmax+1):
        for l2 in range(lmax+1):
            if abs(l2 - l1) > L or l2 + l1 < L:
                continue
            rho1ijplambda[:,:,:,:,il] = cg.combine_einsum(rhoi[...,lm_slice(l1)], gij[...,lm_slice(l2)],
                                                L, combination_string="jn,ijN->ijnN")  # <-- this should be the only difference with the rho1j_lambda implementation!!!
            parity[il] *= (1-2*(l1%2)) * (1-2*(l2%2))
            il+=1
    return rho1ijplambda, parity

def compute_all_rho1ijp_lambda(rhoi, gij, cg, rhoijp_pca=None):
    lmax = int(np.sqrt(rhoi.shape[-1])) -1
    rhoijp = {}
    prhoijp = {}
    for sL in range(lmax+1):
        rhoijp[sL], prhoijp[sL] = compute_rho1ijp_lambda(rhoi, gij, sL, cg)
        if rhoijp_pca is not None:
            rhoijp[sL], prhoijp[sL] = apply_rhoij_pca(rhoijp[sL], prhoijp[sL], rhoijp_pca)
    return rhoijp, prhoijp


def compute_rho11ijp_lambda(rhoi, rhoij_l, L, cg, prfeats):
    """ computes |rho^11p_{ij}; lm>, i.e. the pair term decorated with the description of the j-atom AND the i-atom! """

    lmax = int(np.sqrt(rhoi.shape[-1])) -1

    # "flatten" rhoij_l
    nrhoij = {}
    for l2 in range(lmax+1):
        nrhoij[l2] = rhoij_l[l2].reshape(rhoij_l[l2].shape[:2]+(-1,)+rhoij_l[l2].shape[-2:])
    rhoij_l = nrhoij

    # "flatten" rhoi
    rhoi = rhoi.reshape((rhoi.shape[0], -1, rhoi.shape[-1]))

    # can't work out analytically how many terms we have, so we precompute it here
    nl = 0
    for l1 in range(lmax+1):
        for l2 in range(lmax+1):   # can't use symmetry when combining rhoi and rhoij features
            if abs(l2 - l1) > L or l2 + l1 < L:
                continue
            nl += rhoij_l[l1].shape[-2]   # l-soap indices

    # natom, natom, nel, nmax, nrhoij, nl, M
    # if we do PCA, all the rho2 indices are collapsed
    shape = rhoij_l[0].shape[:3] + (rhoi.shape[1], nl, 2*L+1)

    rho11ijp = np.zeros(shape)
    prho11ijp = np.zeros(nl)

    jl = 0
    for l1 in range(lmax+1):
        n2j = rhoij_l[l1].shape[-2]
        for l2 in range(lmax+1):   # can't use symmetry when combining rhoi and rhoij features
            if abs(l2 - l1) > L or l2 + l1 < L:
                continue
            rho11ijp[...,jl:jl+n2j,:] = cg.combine_einsum(rhoij_l[l1][...], rhoi[...,lm_slice(l2)],
                                                L, combination_string="ijkf,iq->ijkqf")
            prho11ijp[jl:jl+n2j] = prfeats[l1] * (1-2*(l1%2)) * (1-2*(l2%2))
            jl += n2j
    prho11ijp *= (1-2*(L%2))
    return rho11ijp, prho11ijp

def compute_rho2ij_lambda(rho2i_l, gij, L, cg, prho2i):
    """ computes |rho^2_{ij}; lm> """

    lmax = int(np.sqrt(gij.shape[-1])) - 1

    nl = 0
    for l1 in range(lmax+1):
        for l2 in range(lmax+1):   # can't use symmetry when combining rhoi and rhoij features
            if abs(l2 - l1) > L or l2 + l1 < L:
                continue
            nl += rho2i_l[l1].shape[-2]   # l-soap indices

    # natom, natom, nel, nmax, nmax, nl, M
    # if we do PCA, all the rho2 indices are collapsed
    shape = gij.shape[:3] + (
            rho2i_l[0].shape[1:5] if len(rho2i_l[0].shape)>4 else () ) + (
             nl,
             2*L+1)

    rho2ij = np.zeros(shape)
    prho2ij = np.zeros(nl)

    il = 0
    for l1 in range(lmax+1):
        n2i = rho2i_l[l1].shape[-2]
        for l2 in range(lmax+1):   # can't use symmetry when combining rhoi and rhoij features
            if abs(l2 - l1) > L or l2 + l1 < L:
                continue
            rho2ij[...,il:il+n2i,:] = cg.combine_einsum(rho2i_l[l1], gij[...,lm_slice(l2)],
                                                L, combination_string="i...k,ijP->ijP...k")
            prho2ij[il:il+n2i] = prho2i[l1] * (1-2*(l1%2)) * (1-2*(l2%2))
            il += n2i
    prho2ij *= (1-2*(L%2))
    return rho2ij, prho2ij


def compute_rhoi_pca(rhoi, npca):
    """ computes PCA contraction with combined elemental and radial channels.
    returns the contraction matrices
    """

    # compresses further the spherical expansion features across species
    pca_vh = []
    s_sph = []
    lmax = int(np.sqrt(rhoi.shape[-1]))-1
    for l in range(lmax+1):
        xl = np.moveaxis(rhoi[:,:,:,lm_slice(l)],3,1).reshape((rhoi.shape[0]*(2*l+1),-1))
        u, s, vt = sp.sparse.linalg.svds(xl, k=(min(xl.shape)-1), return_singular_vectors='vh')
        s_sph.append(s[::-1])
        pca_vh.append(vt[-npca:][::-1].T)
    return pca_vh, s_sph

def apply_rhoi_pca(rhoi, pca):
    npca = pca[0].shape[-1]
    crhoi = np.zeros((rhoi.shape[0],1,npca,rhoi.shape[-1]))
    lmax = int(np.sqrt(rhoi.shape[-1]))-1
    for l in range(lmax+1):
        xl = np.moveaxis(rhoi[:,:,:,lm_slice(l)],-1,1).reshape((rhoi.shape[0]*(2*l+1),-1))
        crhoi[...,lm_slice(l)] = np.moveaxis((xl@pca[l]).reshape(rhoi.shape[0],2*l+1,1,npca),1,-1)
    return crhoi

def compute_all_rho1i_lambda(rhoi, cg, rhoi_pca=None):
    lmax = int(np.sqrt(rhoi.shape[-1])) -1
    if rhoi_pca is None:
        rhoi = rhoi.copy().reshape((rhoi.shape[0],-1, rhoi.shape[-1]))
    else:
        rhoi = apply_rhoi_pca(rhoi, rhoi_pca)
    return rhoi, None

def compute_rho2i_pca(rhoi, cg, npca, progress = (lambda x:x)):
    """ computes PCA contraction with combined elemental and radial channels.
    returns the contraction matrices
    """

    # compresses across ALL q-channels, but treat separately the symmetry l-channels and the parity
    pca_vh = {}
    s_rho2i = {}
    lmax = int(np.sqrt(rhoi.shape[-1]))-1
    for l in progress(range(lmax+1)):
        rho2il, prho2il = compute_rho2i_lambda(rhoi, l, cg)
        for pi in [-1,1]:
            ipi = np.where(prho2il==pi)[0]
            if len(ipi) ==0:
                continue
            xl = np.moveaxis(rho2il[...,ipi,:],-1,1).reshape((rho2il.shape[0]*(2*l+1),-1))
            u, s, vt = sp.sparse.linalg.svds(xl, k=(min(min(xl.shape)-1,npca*2)), return_singular_vectors='vh')
            pca_vh[(l,pi)]=vt[-npca:][::-1].T
            s_rho2i[(l,pi)] = s[::-1]

    return pca_vh, s_rho2i

def apply_rho2i_pca(rho2i, prho2i, pca):
    l = (rho2i.shape[-1]-1)//2
    cxl = []
    pxl = []
    for pi in [-1,1]:
        ipi = np.where(prho2i==pi)[0]
        if len(ipi) ==0:
            continue
        xl = np.moveaxis(rho2i[...,ipi,:],-1,1).reshape((rho2i.shape[0]*(2*l+1),-1))
        cxl.append(xl @ pca[(l,pi)])
        pxl.append(pi*np.ones(cxl[-1].shape[-1],dtype=int))
    cxl = np.moveaxis(np.concatenate(cxl, axis=1).reshape(rho2i.shape[0],(2*l+1),-1),1,2)
    pxl = np.hstack(pxl)
    return cxl, pxl

def compute_all_rho2i_lambda(rhoi, cg, rho2i_pca=None):
    lmax = int(np.sqrt(rhoi.shape[-1])) -1
    rho2i = {}
    prho2i = {}
    for sL in range(lmax+1):
        rho2i[sL], prho2i[sL] = compute_rho2i_lambda(rhoi, sL, cg)
        if rho2i_pca is not None:
            rho2i[sL], prho2i[sL] = apply_rho2i_pca(rho2i[sL], prho2i[sL], rho2i_pca)
    return rho2i, prho2i

def compute_rhoij_pca(frames, hypers, cg, nu, npca, rho1i_pca=None, rho2i_pca=None, rhoij_pca=None, lmax=None, mp_feats = False, progress = (lambda x:x)):
    """ computes PCA contraction for pair features. do one frame at a time because of memory """

    spex = SphericalExpansion(**hypers)
    hypers_ij = deepcopy(hypers)
    hypers_ij["expansion_by_species_method"] = "structure wise"
    spex_ij = SphericalExpansion(**hypers_ij)

    if lmax is None:
        lmax = hypers["max_angular"]
    pca = {}
    s_pca = {}
    cov = {}
    for f in progress(frames):
        frhoi = compute_rhoi(f, spex, hypers)
        fgij = compute_gij(f, spex_ij, hypers_ij)
        rhonui, prhonui = compute_all_rho1i_lambda(frhoi, cg, rho1i_pca)
        if nu > 1:
            if mp_feats:
                rhonuij, prhonuij = compute_all_rho1ijp_lambda(rhonui, fgij, cg, rhoij_pca)
            else:
                rhonui, prhonui = compute_all_rho2i_lambda(rhonui, cg, rho2i_pca)

        for l in range(lmax+1):
            if nu==1:
                if mp_feats:
                    lrhoij, prhoij = compute_rho1ijp_lambda(rhonui, fgij, l, cg, prhonui)
                else:
                    lrhoij, prhoij = compute_rho1ij_lambda(rhonui, fgij, l, cg, prhonui)
            else:
                if mp_feats:
                    lrhoij, prhoij = compute_rho11ijp_lambda(rhonui, rhonuij, l, cg, prhonuij)
                else:
                    lrhoij, prhoij = compute_rho2ij_lambda(rhonui, fgij, l, cg, prhonui)
            for pi in [-1,1]:
                ipi = np.where(prhoij==pi)[0]
                if len(ipi) ==0:
                    continue
                xl = np.moveaxis(lrhoij[...,ipi,:],-1,1).reshape((lrhoij.shape[0]*lrhoij.shape[1]*(2*l+1),-1))
                if not (l,pi) in cov:
                    cov[(l,pi)] = xl.T@xl
                else:
                    cov[(l,pi)] += xl.T@xl
    for k in progress(list(cov.keys())):
        u, s, vt = sp.sparse.linalg.svds(cov[k], k=min(npca*2, len(cov[k])-1), return_singular_vectors='vh')
        s_pca[k]=s[::-1]
        pca[k] = vt[-npca:][::-1].T
    return pca, s_pca

def apply_rhoij_pca(rhoij, prhoij, pca):
    l = (rhoij.shape[-1]-1)//2
    cxl = []
    pxl = []
    for pi in [-1,1]:
        ipi = np.where(prhoij==pi)[0]
        if len(ipi) ==0:
            continue
        xl = np.moveaxis(rhoij[...,ipi,:],-1,2).reshape((rhoij.shape[0]*rhoij.shape[1]*(2*l+1),-1))
        cxl.append(xl @ pca[(l,pi)])
        pxl.append(pi*np.ones(cxl[-1].shape[-1],dtype=int))
    cxl = np.moveaxis(np.concatenate(cxl, axis=1).reshape(rhoij.shape[0],rhoij.shape[1],(2*l+1),-1),2,-1)
    pxl = np.hstack(pxl)
    return cxl, pxl


def contract_rhoij(rhoij, prhoij, symbols, elements):
    """ Contracts |rho_ij> over j, creating separate channels for the different species """

    shape = rhoij.shape
    shape = (shape[0], len(elements)) + shape[2:]
    rhoi = np.zeros(shape)
    for ie, e in enumerate(elements):
        we = np.where(symbols==e)[0]
        rhoi[:,ie] = rhoij[:,we].sum(axis=1)
    
    return rhoi.reshape( (shape[0], shape[1]*shape[2])+shape[3:]  ), prhoij
