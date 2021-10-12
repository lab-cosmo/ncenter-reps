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
    # natom, natom, nel, nmax, nmax, lmax+1
    parity = np.ones(rhoi.shape[:-2])
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

def compute_rho0ij_lambda(rhoi, gij, L, cg):
    """ computes |rho^0_{ij}; lm> """
    return gij[..., lm_slice(L)]

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

    # natom, natom, nel, nmax, nmax, lmax+1, lmax+1, M
    shape = (rhoi.shape[0], rhoi.shape[0],
             rhoi.shape[1], rhoi.shape[2],
             gij.shape[2],
             nl,
             2*L+1)
    rho1ijlambda = np.zeros(shape)
    parity = np.ones(nl, dtype = int)*(1-2*(L%2))

    il = 0
    for l1 in range(lmax+1):
        for l2 in range(lmax+1):
            if abs(l2 - l1) > L or l2 + l1 < L:
                continue
            rho1ijlambda[:,:,:,:,:,il] = cg.combine_einsum(rhoi[...,lm_slice(l1)], gij[...,lm_slice(l2)],
                                                L, combination_string="ian,ijN->ijanN")
            parity[il] *= (1-2*(l1%2)) * (1-2*(l2%2))
            il+=1

    return rho1ijlambda, parity

def compute_rho2ij_lambda(rho2i_l, gij, L, cg, prho2i): #, rho2i_pca=None):
    """ computes |rho^1_{ij}; lm> """

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

def compute_rho2i_pca(rhoi, cg, npca, progress = (lambda x:x)):
    """ computes PCA contraction with combined elemental and radial channels.
    returns the contraction matrices
    """

    # compresses across ALL q-channels, but treat separately the symmetry l-channels and the parity
    pca_vh = {}
    s_rho2i = {}
    lmax = int(np.sqrt(rhoi.shape[-1]))-1
    for l in progress(range(lmax+1)):
        rho2il, prho2il = mk_rho2ilambda_fast(rhoi, l, cg)
        for pi in [-1,1]:
            ipi = np.where(prho2il==pi)[0]
            if len(ipi) ==0:
                continue
            xl = np.moveaxis(rho2il[...,ipi,:],-1,1).reshape((rho2il.shape[0]*(2*l+1),-1))
            u, s, vt = sp.sparse.linalg.svds(xl, k=(min(min(xl.shape)-1,npca*2)), return_singular_vectors='vh')
            pca_vh[(l,pi)]=vt[-npca:][::-1].T
            s_rho2i[(l,pi)] = s[::-1]

    return pca_vh, s_rho2i

def rho2i_pca_transform(rho2i, prho2i, pca):
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
            rho2i[sL], prho2i[sL] = rho2i_pca_transform(rho2i[sL], prho2i[sL], rho2i_pca)
    return rho2i, prho2i

def do_rhoij_pca(frames, hypers, cg, rhoij_func, sph_pca, rho2i_pca, npca, lmax=None, progress = (lambda x:x)):
    """ computes PCA contraction for pair features
    """

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
        fsph = spherical_expansion_reshape(spex.transform(f).get_features(spex), **hypers)
        fsph = sph_pca_transform(fsph, sph_pca)
        fgij = get_gij_fast(f, spex_ij, hypers_ij)
        if rhoij_func.__name__ == "mk_rho1ijlambda_fast":
            rhoi, prhoi = fsph, None
        else:
            rhoi, prhoi = mk_rho2ilambda_full(fsph, cg, rho2i_pca)

        for l in range(lmax+1):
            lrhoij, prhoij = rhoij_func(rhoi, fgij, l, cg, prhoi)
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

def rhoij_pca_transform(rhoij, prhoij, pca):
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

def do_full_features(frames, orbs, hypers, lmax, cg, scale=1, select_feats = None, half_hete = True,
                     sph_pca = None, rho2i_pca = None,
                     rhoij_func = compute_rho1ij_lambda, rhoij_rho2i_pca = None, rhoij_pca = None,
                     verbose = False
                     ):
    """
        Computes the full set of features needed to learn matrix elements up to lmax.
        Options are fluid, but here are some that need an explanation:

        select_feats = dict(type=["diag", "offd_m", "offd_p", "hete"], block = ('el1', ['el2',] L, pi) )
        does the minimal amount of calculation to evaluate the selected block. other terms might be computed as well if they come for free.
    """

    spex = SphericalExpansion(**hypers)
    sph = spherical_expansion_reshape(spex.transform(frames).get_features(spex), **hypers)

    # compresses further the spherical expansion features across species
    if sph_pca is not None:
        #npca = sph_pca.shape[-1]
        #csph = np.zeros((sph.shape[0],1,npca,sph.shape[-1]))
        #for l in range(hypers["max_angular"]+1):
        #    xl = np.moveaxis(sph[:,:,:,lm_slice(l)],3,1).reshape((sph.shape[0]*(2*l+1),-1))
        #    csph[...,lm_slice(l)] = np.moveaxis((xl@sph_pca[l]).reshape(sph.shape[0],2*l+1,1,npca),1,3)
        sph = sph_pca_transform(sph, sph_pca)
        #print(sph.shape, csph.shape)

    # makes sure that the spex used for the pair terms uses adaptive species
    hypers_ij = deepcopy(hypers)
    hypers_ij["expansion_by_species_method"] = "structure wise"
    spex_ij = SphericalExpansion(**hypers_ij)

    tnat = 0
    els = list(orbs.keys())
    nel = len(els)
    # prepare storage
    elL = list(itertools.product(els,range(lmax+1),[-1,1]))
    hetL = [ (els[i1], els[i2], L, pi) for i1 in range(nel) for i2 in range((i1+1 if half_hete else 0), nel) for L in range(lmax+1) for pi in [-1,1] ]
    feats = dict(diag = { L: [] for L in elL },
                 offd_p = { L: [] for L in elL },
                 offd_m = { L: [] for L in elL },
                 hete =   { L: [] for L in hetL },)

    if rhoij_rho2i_pca is None and rho2i_pca is not None:
        rhoij_rho2i_pca = rho2i_pca

    #before = tracemalloc.take_snapshot()
    for f in frames:
        fnat = len(f.numbers)
        fsph = sph[tnat:tnat+fnat]*scale
        fgij = compute_gij(f, spex_ij, hypers_ij)*scale

        if (select_feats is None or select_feats["type"]!="diag") and rhoij_func.__name__ == "compute_rho2ij_lambda":
            rhoi_full, prhoi_full = compute_all_rho2i_lambda(fsph, cg, rhoij_rho2i_pca)
        else:
            rhoi_full, prhoi_full = fsph, None

        for L in range(lmax+1):
            if select_feats is not None and L>0 and select_feats["block"][-2] != L:
                continue
            lrho2, prho2 = compute_rho2i_lambda(fsph, L, cg)
            if rho2i_pca is not None:
                lrho2, prho2 = rho2i_pca_transform(lrho2, prho2, rho2i_pca)

            if select_feats is None or select_feats["type"]!="diag":
                lrhoij, prhoij = rhoij_func(rhoi_full, fgij, L, cg, prhoi_full)
                if rhoij_pca is not None:
                    lrhoij, prhoij = rhoij_pca_transform(lrhoij, prhoij, rhoij_pca)

            for i, el in enumerate(els):
                iel = np.where(f.symbols==el)[0]
                if len(iel) == 0:
                    continue
                if select_feats is not None and el != select_feats["block"][0]:
                    continue

                for pi in [-1,1]:
                    wherepi = np.where(prho2==pi)[0]
                    if len(wherepi)==0: continue
                    feats['diag'][(el, L, pi)].append(lrho2[...,wherepi,:][iel].reshape((len(iel), -1, 2*L+1) ) )

                if select_feats is not None and select_feats["type"]=="diag":
                    continue

                triu = np.triu_indices(len(iel), 1)
                ij_up = (iel[triu[0]],iel[triu[1]]) # ij indices, i>j
                ij_lw = (ij_up[1], ij_up[0]) # ij indices, i<j
                lrhoij_p = (lrhoij[ij_up] + lrhoij[ij_lw])/np.sqrt(2)
                lrhoij_m = (lrhoij[ij_up] - lrhoij[ij_lw])/np.sqrt(2)
                for pi in [-1,1]:
                    wherepi = np.where(prhoij==pi)[0];
                    if len(wherepi)==0 or len(ij_up[0])==0: continue
                    feats['offd_p'][(el, L, pi)].append(lrhoij_p[...,wherepi,:].reshape(lrhoij_p.shape[0], -1, 2*L+1))
                    feats['offd_m'][(el, L, pi)].append(lrhoij_m[...,wherepi,:].reshape(lrhoij_m.shape[0], -1, 2*L+1))

                if select_feats is not None and select_feats["type"]!="hete":
                    continue
                for elb in els[i+1:]:
                    ielb = np.where(f.symbols==elb)[0]
                    if len(ielb) == 0:
                        continue
                    if select_feats is not None and elb != select_feats["block"][1]:
                        continue

                    # combines rho_ij and rho_ji
                    lrhoij_het = lrhoij[iel][:,ielb]
                    lrhoij_het_rev = np.swapaxes(lrhoij[ielb][:,iel],1,0)
                    # make a copy and not a slice, so we keep better track
                    for pi in [-1,1]:
                        wherepi = np.where(prhoij==pi)[0];
                        if len(wherepi)==0:
                            continue
                        lrhoij_het_pi = lrhoij_het[...,wherepi,:]
                        lrhoij_het_rev_pi = lrhoij_het_rev[...,wherepi,:]
                        feats['hete'][(el, elb, L, pi)].append(
                            np.concatenate([
                            lrhoij_het_pi.reshape(
                                (lrhoij_het.shape[0]*lrhoij_het.shape[1],-1,2*L+1) )
                            ,
                            lrhoij_het_rev_pi.reshape(
                                (lrhoij_het_rev.shape[0]*lrhoij_het_rev.shape[1],-1,2*L+1) )
                            ], axis=-2)
                        )
                    #del(lrhoij_het)
                #del(lrhoij_p, lrhoij_m)
            #del(lrhoij, lrho2)
        tnat+=fnat

    #mid = tracemalloc.take_snapshot()
    #top_stats = mid.compare_to(before, 'lineno')
    #print("[ Top 10 differences ]")
    #for stat in top_stats[:10]:  print(stat)


    # cleans up combining frames blocks into single vectors - splitting also odd and even blocks
    if verbose: print("combining", get_size(feats))
    for k in feats.keys():
        for b in list(feats[k].keys()):
            if len(feats[k][b]) == 0:
                continue
            block = np.vstack(feats[k][b])
            feats[k].pop(b)
            if len(block) == 0:
                continue

            feats[k][b] = block.reshape((block.shape[0], -1, 1+2*b[-2]))

    if verbose: print("compare ", get_size(feats))
    if verbose: print("done", gc.collect())
    #then = tracemalloc.take_snapshot()
    #top_stats = then.compare_to(mid, 'lineno')
    #print("[ Top 10 differences ]")
    #for stat in top_stats[:10]:  print(stat)
    return feats

