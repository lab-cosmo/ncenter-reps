import numpy as np


###########  I/O UTILITIES ##############
def fix_pyscf_l1(fock, frame, orbs):
    """ pyscf stores l=1 terms in a xyz order, corresponding to (m=1, 0, -1).
        this converts into a canonical form where m is sorted as (-1, 0,1) """
    idx = []
    iorb = 0;
    atoms = list(frame.symbols)
    for atype in atoms:
        cur=()
        for ia, a in enumerate(orbs[atype]):
            n,l,m = a
            if (n,l) != cur:
                if l == 1:
                    idx += [iorb+1, iorb+2, iorb]
                else:
                    idx += range(iorb, iorb+2*l+1)
                iorb += 2*l+1
                cur = (n,l)
    return fock[idx][:,idx]

########### HAMILTONIAN MANIPULATION ###########
def lowdin_orthogonalize(fock, s):
    """
    lowdin orthogonalization of a fock matrix computing the square root of the overlap matrix
    """
    eva, eve = np.linalg.eigh(s)
    sm12 = eve @ np.diag(1.0/np.sqrt(eva)) @ eve.T
    return sm12 @ fock @ sm12


########## ORBITAL INDEXING ##############
def orbs_base(orbs):
    # converts list of orbitals into an index to access different sub-blocks
    norbs = 0
    io_base = {}
    el_dict = {}
    for el in orbs.keys():
        io_base[el] = norbs
        cur_a = ()
        for na, la, ma in orbs[el]:
            if cur_a == (na,la): continue
            cur_a = (na, la)
            el_dict[(na+io_base[el], la)] = el
            norbs+=1

    return io_base, el_dict

############ matrix/block manipulations ###############
def matrix_to_blocks(fock, frame, orbs):
    """ splits an atomic orbital matrix to (uncoupled momentum) orbital blocks. """
    # maps atom types to different n indices
    io_base, _ = orbs_base(orbs)

    # prepares storage
    diaglist = {}
    offdlist_p = {}
    offdlist_m = {}
    heterolist = {}

    # creates storage. these are the blocks of the matrix we'll have to fill up later
    lorbs = []
    for el_a in orbs.keys():
        for ia, a in enumerate(orbs[el_a]):
            na, la, ma = a
            na += io_base[el_a] # adds element offset
            for el_b in orbs.keys():
                for ib, b in enumerate(orbs[el_b]):
                    nb, lb, mb = b
                    nb += io_base[el_b] # adds element offset
                    if ( (nb>na or (nb==na and lb>=la)) and
                        not (na,la,nb,lb) in lorbs ):
                        orb = (na,la,nb,lb)
                        lorbs.append(orb)
                        if el_a == el_b:
                            diaglist[orb] = []
                            offdlist_p[orb] = []
                            offdlist_m[orb] = []
                        else:
                            heterolist[orb] = []


    # reads in and partitions into blocks
    ki = 0
    nat = len(frame.numbers)
    for i in range(nat):
        el_a = frame.symbols[i]
        cur_a = ()
        for ia, oa in enumerate(orbs[el_a]):
            na, la, ma = oa
            na += io_base[el_a]
            # we read the Hamiltonian in blocks
            if (cur_a == (na,la)): continue
            cur_a = (na,la)
            kj = 0
            for j in range(nat):
                el_b = frame.symbols[j]
                cur_b = ()
                for ib, ob in enumerate(orbs[el_b]):
                    nb, lb, mb = ob
                    nb += io_base[el_b] # adds element offset
                    if (cur_b == (nb,lb)): continue  # only read at the beginning of each m block
                    cur_b = (nb,lb)
                    if (nb<na or (nb==na and lb<la)): continue
                    orb = (na,la,nb,lb)
                    blockij = fock[ki+ia:ki+ia+2*la+1, kj+ib:kj+ib+2*lb+1]
                    if (i==j):
                        diaglist[orb].append(blockij)
                    elif (i<j and el_a == el_b):
                        blockji= fock[kj+ia:kj+ia+2*la+1, ki+ib:ki+ib+2*lb+1]
                        offdlist_p[orb].append((blockij+blockji)/np.sqrt(2))
                        offdlist_m[orb].append((blockij-blockji)/np.sqrt(2))
                    elif(el_a != el_b):
                        heterolist[orb].append(blockij)
                kj += len(orbs[el_b])
        ki += len(orbs[el_a])

    # stores as ndarray for more flexible indexing
    for orb in lorbs:
        for d in [diaglist, offdlist_p, offdlist_m, heterolist]:
            if orb in d:
                d[orb] = np.asarray(d[orb])

    return dict( diag=diaglist, offd_p=offdlist_p, offd_m=offdlist_m, hete=heterolist)

def matrix_to_ij_indices(fock, frame, orbs):
    """ Creates indices to the atoms involved in each block """
    # maps atom types to different n indices
    io_base, _ = orbs_base(orbs)

    # prepares storage
    diaglist = {}
    offdlist_p = {}
    offdlist_m = {}
    heterolist = {}

    # creates storage. these are the blocks of the matrix we'll have to fill up later
    lorbs = []
    for el_a in orbs.keys():
        for ia, a in enumerate(orbs[el_a]):
            na, la, ma = a
            na += io_base[el_a] # adds element offset
            for el_b in orbs.keys():
                for ib, b in enumerate(orbs[el_b]):
                    nb, lb, mb = b
                    nb += io_base[el_b] # adds element offset
                    if ( (nb>na or (nb==na and lb>=la)) and
                        not (na,la,nb,lb) in lorbs ):
                        orb = (na,la,nb,lb)
                        lorbs.append(orb)
                        if el_a == el_b:
                            diaglist[orb] = []
                            offdlist_p[orb] = []
                            offdlist_m[orb] = []
                        else:
                            heterolist[orb] = []


    # reads in and partitions into blocks
    ki = 0
    nat = len(frame.numbers)
    for i in range(nat):
        el_a = frame.symbols[i]
        cur_a = ()
        for ia, oa in enumerate(orbs[el_a]):
            na, la, ma = oa
            na += io_base[el_a]
            # we read the Hamiltonian in blocks
            if (cur_a == (na,la)): continue
            cur_a = (na,la)
            kj = 0
            for j in range(nat):
                el_b = frame.symbols[j]
                cur_b = ()
                for ib, ob in enumerate(orbs[el_b]):
                    nb, lb, mb = ob
                    nb += io_base[el_b] # adds element offset
                    if (cur_b == (nb,lb)): continue  # only read at the beginning of each m block
                    cur_b = (nb,lb)
                    if (nb<na or (nb==na and lb<la)): continue
                    orb = (na,la,nb,lb)
                    blockij = (i,j)
                    if (i==j):
                        diaglist[orb].append(blockij)
                    elif (i<j and el_a == el_b):
                        offdlist_p[orb].append(blockij)
                        offdlist_m[orb].append(blockij)
                    elif(el_a != el_b):
                        heterolist[orb].append(blockij)
                kj += len(orbs[el_b])
        ki += len(orbs[el_a])

    # stores as ndarray for more flexible indexing
    for orb in lorbs:
        for d in [diaglist, offdlist_p, offdlist_m, heterolist]:
            if orb in d:
                d[orb] = np.asarray(d[orb])

    return dict( diag=diaglist, offd_p=offdlist_p, offd_m=offdlist_m, hete=heterolist)

def blocks_to_matrix(blocks, frame, orbs):
    """ assembles (uncoupled momentum) orbital blocks into a matrix form
    NB - the l terms are stored in canonical order, m=-l..l """

    io_base, _ = orbs_base(orbs)
    norbs = 0
    for el in list(frame.symbols):
        norbs+= len(orbs[el])
    nat = len(list(frame.symbols))
    unfock = np.zeros((norbs, norbs))

    bidx = {}
    for k in blocks.keys():
        bidx[k] = {}
        for bk in blocks[k].keys():
            bidx[k][bk] = 0
    cur_a = ()
    ki = 0
    nat = len(frame.numbers)
    for i in range(nat):
        el_a = frame.symbols[i]
        cur_a = ()
        for ia, oa in enumerate(orbs[el_a]):
            na, la, ma = oa
            na += io_base[el_a]
            # we read the Hamiltonian in blocks
            if (cur_a == (na,la)): continue
            cur_a = (na,la)
            kj = 0
            for j in range(nat):
                el_b = frame.symbols[j]
                cur_b = ()
                for ib, ob in enumerate(orbs[el_b]):
                    nb, lb, mb = ob
                    nb += io_base[el_b] # adds element offset
                    if (cur_b == (nb,lb)): continue  # only read at the beginning of each m block
                    cur_b = (nb,lb)
                    if (nb<na or (nb==na and lb<la)): continue
                    orb = (na,la,nb,lb)
                    if (i==j):
                        blockij = blocks['diag'][orb][bidx['diag'][orb]]
                        unfock[ki+ia:ki+ia+2*la+1, kj+ib:kj+ib+2*lb+1] = blockij
                        unfock[ki+ib:ki+ib+2*lb+1, kj+ia:kj+ia+2*la+1] = blockij.T
                        bidx['diag'][orb] += 1
                    elif (el_a == el_b and i<j):
                        blockij = (blocks['offd_p'][orb][bidx['offd_p'][orb]]
                                   +blocks['offd_m'][orb][bidx['offd_m'][orb]])/np.sqrt(2)
                        blockji = (blocks['offd_p'][orb][bidx['offd_p'][orb]]
                                   -blocks['offd_m'][orb][bidx['offd_m'][orb]])/np.sqrt(2)
                        unfock[ki+ia:ki+ia+2*la+1, kj+ib:kj+ib+2*lb+1] = blockij
                        unfock[kj+ib:kj+ib+2*lb+1, ki+ia:ki+ia+2*la+1] = blockij.T
                        unfock[kj+ia:kj+ia+2*la+1, ki+ib:ki+ib+2*lb+1] = blockji
                        unfock[ki+ib:ki+ib+2*lb+1, kj+ia:kj+ia+2*la+1] = blockji.T
                        bidx['offd_p'][orb] += 1
                        bidx['offd_m'][orb] += 1
                    elif (el_a != el_b):
                        blockij = blocks['hete'][orb][bidx['hete'][orb]]
                        unfock[ki+ia:ki+ia+2*la+1, kj+ib:kj+ib+2*lb+1] = blockij
                        unfock[kj+ib:kj+ib+2*lb+1, ki+ia:ki+ia+2*la+1] = blockij.T
                        bidx['hete'][orb] += 1
                kj += len(orbs[el_b])
        ki += len(orbs[el_a])
    return unfock

def couple_blocks(dcoef, cg):
    """ converts coefficients (fock matrix blocks) from uncoupled to coupled form """
    dccoef = {}
    for dk in dcoef.keys():
        dccoef[dk] = {}
        for k in dcoef[dk].keys():
            if len(dcoef[dk][k]) > 0:
                # computes the coupled representation
                coupled = [ next(iter(cg.couple(el).values())) for el in dcoef[dk][k] ]
                # creates the dictionary
                dccoef[dk][k] = {}
                for L in coupled[0].keys():
                    dccoef[dk][k][L] = np.asarray([el[L] for el in coupled])
    return dccoef

def decouple_blocks(dccoef, cg):
    """ converts coefficients (fock matrix blocks) from coupled to uncoupled form """
    dcoef = {}
    for dk in dccoef.keys():
        dcoef[dk] = {}
        for k in dccoef[dk].keys():
            decoup = []
            if len(dccoef[dk][k])==0:
                continue
            nitems = len( list(dccoef[dk][k].values())[0])
            for i in range(nitems):
                coupled = {(k[1], k[3]): {}}
                for L in dccoef[dk][k].keys():
                    coupled[(k[1], k[3])][L] = dccoef[dk][k][L][i]
                decoup.append(cg.decouple(coupled))
            dcoef[dk][k] = np.asarray(decoup)
    return dcoef

def matrix_list_to_blocks(focks, frames, orbs, cg, progress = (lambda x: x)):
    """ Computes coupled-momemtum blocks of the matrices for a bunch of frames,
    collecting all the learning targets in a single list. Also returns indices
    that allows extracting the blocks from each matrix. """

    blocks = couple_blocks(matrix_to_blocks(focks[0], frames[0], orbs), cg)
    slices = [{}]
    for k in blocks.keys():
        slices[0][k] = {}
        for orb in blocks[k]:
            L0 = list(blocks[k][orb].keys())[0]
            slices[0][k][orb] = slice(0, len(blocks[k][orb][L0]))
    for ifr in progress(range(1,len(frames))):
        fc = couple_blocks(matrix_to_blocks(focks[ifr], frames[ifr], orbs), cg)
        slices.append({})
        for k in fc.keys():
            slices[-1][k] = {}
            for orb in fc[k]:
                L0 = list(fc[k][orb].keys())[0]
                if not orb in blocks[k]:
                    # extend the blocks if more orbital combinations appear
                    blocks[k][orb] = fc[k][orb]
                    slices[-1][k][orb] = slice(0, len(blocks[k][orb][L0]))
                else:
                    slices[-1][k][orb] = slice(len(blocks[k][orb][L0]),
                                               len(blocks[k][orb][L0])+len(fc[k][orb][L0]) )
                    for L in fc[k][orb]:
                        blocks[k][orb][L] = np.vstack([blocks[k][orb][L], fc[k][orb][L] ] )
    return blocks, slices

def get_block_idx(frame_idx, slices):
    """ returns a dictionary with the indices associated with
        the hamiltonians of the frames in frame_idx """
    idx_slices = { k:{} for k in slices[0].keys() }
    for i in frame_idx:
        for k in slices[i].keys():
            for b in slices[i][k].keys():
                if not b in idx_slices[k]:
                    idx_slices[k][b] = []
                idx_slices[k][b] += list(range( slices[i][k][b].start, slices[i][k][b].stop))
    return idx_slices

def coupled_block_slice(dccoef, slices):
    dcoef = {}
    for dk in dccoef.keys():
        dcoef[dk] = {}
        for orb in dccoef[dk].keys():
            if type(slices) is dict:
                if orb in slices[dk]:
                    sl = slices[dk][orb]
                else:
                    continue
            else:
                sl = slices
            dcoef[dk][orb] = {}
            for L in dccoef[dk][orb]:
                dcoef[dk][orb][L] = dccoef[dk][orb][L][sl]
    return dcoef

def blocks_to_matrix_list(blocks, frames, slices, orbs, cg):
    """ Transforms coupled-momentum blocks to a list of fock matrices. """
    focks = []
    ntot = 0
    for ifr in range(len(slices)):
        dec = decouple_blocks(coupled_block_slice(blocks, slices[ifr]), cg)
        focks.append(blocks_to_matrix(dec, frames[ifr], orbs))
    return focks

def block_to_feat_index(tblock, kblock, lblock, orbs):
    obase = orbs_base(orbs)
    if tblock != "hete":
        fblock = (obase[1][kblock[0:2]], lblock, (1-2*(kblock[1]%2))*(1-2*(kblock[3]%2))*(1-2*(lblock%2)) )
    else:
        fblock = (obase[1][kblock[0:2]], obase[1][kblock[2:4]], lblock, (1-2*(kblock[1]%2))*(1-2*(kblock[3]%2))*(1-2*(lblock%2)))
    return fblock

def merge_blocks(lblocks):
    """ Takes a list of block dictionaries and consolidates into a single list """
    rblocks = {}
    for block in lblocks:
        for k in block:
            if not k in rblocks:
                rblocks[k] = block[k]
            else:
                for b in block[k]:
                    if not b in rblocks[k] or len(rblocks[k][b])==0:
                        rblocks[k][b] = block[k][b]
                    elif len(block[k][b])>0:
                        rblocks[k][b] = np.concatenate([rblocks[k][b], block[k][b]], axis=0)
    return rblocks

def slice_fraction(blocks, tf = 0.5):
    train_slices = {}
    for k in blocks.keys():
        train_slices[k] = {}
        for orb in blocks[k]:
            L0 = list(blocks[k][orb].keys())[0]
            train_slices[k][orb] = slice(0,int(len(blocks[k][orb][L0])*tf))
    return train_slices
