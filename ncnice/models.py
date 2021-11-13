import numpy as np
import sklearn
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.base import RegressorMixin, BaseEstimator
from time import time
import scipy
from .hamiltonians import orbs_base, block_to_feat_index

# avoid polluting output with CV failures
import warnings
warnings.filterwarnings('ignore', category=scipy.linalg.LinAlgWarning)
warnings.filterwarnings('ignore', category=sklearn.exceptions.FitFailedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class SASplitter:
    """ CV splitter that takes into account the presence of "L blocks"
    associated with symmetry-adapted regression. Basically, you can trick conventional
    regression schemes to work on symmetry-adapted data y^M_L(A_i) by having the (2L+1)
    angular channels "unrolled" into a flat array. Then however splitting of train/test
    or cross validation must not "cut" across the M block. This takes care of that.
    """
    def __init__(self, L, cv=2):
        self.L = L
        self.cv = cv
        self.n_splits = cv

    def split(self, X, y, groups=None):

        ntrain = X.shape[0]
        if ntrain % (2*self.L+1) != 0:
            raise ValueError("Size of training data is inconsistent with the L value")
        ntrain = ntrain // (2*self.L+1)
        nbatch = (2*self.L+1)*(ntrain//self.n_splits)
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        for n in range(self.n_splits):
            itest = idx[n*nbatch:(n+1)*nbatch]
            itrain = np.concatenate([idx[:n*nbatch], idx[(n+1)*nbatch:]])
            yield itrain, itest

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

class SARidge(Ridge):
    """ Symmetry-adapted ridge regression class """

    def __init__(self, L, alpha=1, alphas=None, cv=2,
                 fit_intercept=False, scoring='neg_root_mean_squared_error'):
        self.L = L
        # L>0 components have zero mean by symmetry
        if L>0:
            fit_intercept = False
        self.cv = SASplitter(L, cv)
        self.alphas = alphas
        self.cv_stats = None
        self.scoring = scoring
        super(SARidge, self).__init__(alpha=alpha, fit_intercept=fit_intercept)

    def fit(self, Xm, Ym, X0=None):
        # this expects properties in the form [i, m] and features in the form [i, q, m]
        # in order to train a SA-GPR model the m indices have to be moved and merged with the i

        Xm_flat = np.moveaxis(Xm, 2, 1).reshape((-1, Xm.shape[1]))
        Ym_flat = Ym.flatten()
        if self.alphas is not None:
            # determines alpha by grid search
            rcv = Ridge(fit_intercept=self.fit_intercept)
            gscv = GridSearchCV(rcv, dict(alpha=self.alphas), cv=self.cv, scoring=self.scoring)
            gscv.fit(Xm_flat, Ym_flat)
            self.cv_stats = gscv.cv_results_
            self.alpha = gscv.best_params_["alpha"]

        super(SARidge, self).fit(Xm_flat, Ym_flat)
    def predict(self, Xm, X0=None):

        Y = super(SARidge, self).predict(np.moveaxis(Xm, 2, 1).reshape((-1, Xm.shape[1])))
        return Y.reshape((-1, 2*self.L+1))


class SAKernelRidge(KernelRidge):
    """ Symmetry-adapted kernel ridge regression class """

    def __init__(self, L, zeta=[1], zeta_w=None, cv=2, alpha=1, alphas=None,
                 fit_intercept=False, scoring='neg_root_mean_squared_error'):
        self.L = L
        # L>0 components have zero mean by symmetry
        if L>0:
            fit_intercept = False
        self.cv = SASplitter(L, cv)
        if not hasattr(zeta,'__len__'):
            zeta = [zeta]
        if zeta_w is None:
            zeta_w = np.ones(len(zeta))
        self.zeta = zeta
        self.zeta_w = zeta_w

        self.alphas = alphas
        self.scoring = scoring
        self.fit_intercept = fit_intercept
        super(SAKernelRidge, self).__init__(kernel="precomputed", alpha=alpha)

    def fit(self, Xm, Ym, X0):
        # this expects properties in the form [i, m] and features in the form [i, q, m]
        # in order to train a SA-GPR model the m indices have to be moved and merged with the i
        # it also gets *invariant* features to build non-linear sa-gpr kernels

        # computes lambda-soap kernel
        X0_flat = X0.reshape(X0.shape[:2])
        Xm_flat = np.moveaxis(Xm, -1,1).reshape((-1,Xm.shape[1]))
        K0 = X0_flat@X0_flat.T #np.einsum("iq,jq->ij", X0[...,0], X0[...,0])
        KLMM = (Xm_flat@Xm_flat.T).reshape((Xm.shape[0],Xm.shape[-1],Xm.shape[0],Xm.shape[-1]))  #np.einsum("iqm,jqn->imjn", Xm, Xm)
        
        self.KLscale = np.trace(KLMM.reshape( ((2*self.L+1)*len(Xm),((2*self.L+1)*len(Xm))) ))/len(Xm)
        self.K0scale = np.trace(K0)/len(X0)
        if self.K0scale == 0.0 :
            self.K0scale = 1
        if self.KLscale == 0.0 :
            self.KLscale = 1
            
        self.rXm = Xm_flat
        self.rX0 = X0_flat

        Kpoly = KLMM*0.0
        for z, zw in zip(self.zeta, self.zeta_w):
            Kpoly += np.einsum("imjn,ij->imjn",KLMM/self.KLscale, zw*(K0/self.K0scale)**(z-1))

        Kpoly_flat = Kpoly.reshape( ((2*self.L+1)*len(Xm),((2*self.L+1)*len(Xm))) )
        Ym_flat = Ym.flatten()
        self._Y_mean = 0
        if self.fit_intercept:
            self._Y_mean = Ym_flat.mean()
        Ym_flat = Ym_flat - self._Y_mean

        if self.alphas is not None:
            # determines alpha by grid search
            krcv = KernelRidge(kernel="precomputed")
            gscv = GridSearchCV(krcv, dict(alpha=self.alphas), cv=self.cv, scoring=self.scoring)
            gscv.fit(Kpoly_flat, Ym_flat)
            self.cv_stats = gscv.cv_results_
            self.alpha = gscv.best_params_["alpha"]

        super(SAKernelRidge, self).fit( Kpoly_flat, Ym_flat)

    def predict(self, Xm, X0):
        X0_flat = X0.reshape(X0.shape[:2])
        Xm_flat = np.moveaxis(Xm, -1,1).reshape((-1,Xm.shape[1]))
        K0 = X0_flat@self.rX0.T #np.einsum("iq,jq->ij", X0[...,0], X0[...,0])
        KLMM = (Xm_flat@self.rXm.T).reshape((Xm.shape[0],Xm.shape[-1],self.rX0.shape[0],Xm.shape[-1]))  #np.einsum("iqm,jqn->imjn", X

#        K0 = np.einsum("iq,jq->ij", X0[...,0], self.rX0[...,0])
#        KLMM = np.einsum("iqm,jqn->imjn", Xm, self.rXm)
        Kpoly = KLMM*0.0
        for z, zw in zip(self.zeta, self.zeta_w):
            Kpoly += np.einsum("imjn,ij->imjn",KLMM/self.KLscale, zw*(K0/self.K0scale)**(z-1))

        Y = self._Y_mean + super(SAKernelRidge, self).predict(
                      Kpoly.reshape(((2*self.L+1)*len(Xm),-1)))
        return Y.reshape((-1, 2*self.L+1))

class SparseGPRSolver:
    """
    A few quick implementation notes, docs to be done.

    This is meant to solve the sparse GPR problem
    b = (KNM.T@KNM + reg*KMM)^-1 @ KNM.T@y

    The inverse needs to be stabilized with application of a numerical jitter,
    that is expressed as a fraction of the largest eigenvalue of KMM


    """

    def __init__(
        self, KMM, regularizer=1, jitter=0, rkhs_vectors=None, solver="RKHS", relative_jitter=True
    ):

        self.solver = solver
        self.KMM = KMM
        self.relative_jitter = relative_jitter
        self.rkhs_vectors = rkhs_vectors
        self._nM = len(KMM)
        if self.solver == "RKHS" or self.solver == "RKHS-QR":
            start = time()
            if self.rkhs_vectors is None:
                vk, Uk = scipy.linalg.eigh(KMM)
                self.rkhs_vectors = (vk[::-1], Uk[::-1])
            self._vk, self._Uk = self.rkhs_vectors
        elif self.solver == "QR" or self.solver == "Normal":
            # gets maximum eigenvalue of KMM to scale the numerical jitter
            self._KMM_maxeva = scipy.sparse.linalg.eigsh(
                KMM, k=1, return_eigenvectors=False
            )[0]
        else:
            raise ValueError(
                "Solver ",
                solver,
                " not supported. Possible values are [RKHS, RKHS-QR, QR, Normal].",
            )
        if relative_jitter:
            if self.solver == "RKHS" or self.solver == "RKHS-QR":
                self._jitter_scale = self._vk[0]
            elif self.solver == "QR" or self.solver == "Normal":
                self._jitter_scale = self._KMM_maxeva
        else:
            self._jitter_scale = 1.0
        self.set_regularizers(regularizer, jitter)

    def set_regularizers(self, regularizer=1.0, jitter=0.0):
        self.regularizer = regularizer
        self.jitter = jitter
        if self.solver == "RKHS" or self.solver == "RKHS-QR":
            self._nM = len(np.where(self._vk > self.jitter * self._jitter_scale)[0])
            self._PKPhi = self._Uk[:, : self._nM] * 1 / np.sqrt(self._vk[: self._nM])
        elif self.solver == "QR":
            self._VMM = scipy.linalg.cholesky(
                self.regularizer * self.KMM
                + np.eye(self._nM) * self._jitter_scale * self.jitter
            )
        self._Cov = np.zeros((self._nM, self._nM))
        self._KY = None

    def partial_fit(self, KNM, Y, accumulate_only=False):

        if len(Y) > 0:
            # only accumulate if we are passing data
            if len(Y.shape) == 1:
                Y = Y[:, np.newaxis]
            if self.solver == "RKHS":
                Phi = KNM @ self._PKPhi
            elif self.solver == "Normal":
                Phi = KNM
            else:
                raise ValueError(
                    "Partial fit can only be realized with solver = [RKHS, Normal]"
                )
            if self._KY is None:
                self._KY = np.zeros((self._nM, Y.shape[1]))

            self._Cov += Phi.T @ Phi
            self._KY += Phi.T @ Y

        # do actual fit if called with empty array or if asked
        if len(Y) == 0 or (not accumulate_only):
            if self.solver == "RKHS":
                self._weights = self._PKPhi @ scipy.linalg.solve(
                    self._Cov + np.eye(self._nM) * self.regularizer,
                    self._KY,
                    assume_a="pos",
                )
            elif self.solver == "Normal":
                self._weights = scipy.linalg.solve(
                    self._Cov
                    + self.regularizer * self.KMM
                    + np.eye(self.KMM.shape[0]) * self.jitter * self._jitter_scale,
                    self._KY,
                    assume_a="pos",
                )

    def fit(self, KNM, Y):

        if len(Y.shape) == 1:
            Y = Y[:, np.newaxis]
        if self.solver == "RKHS":
            Phi = KNM @ self._PKPhi
            self._weights = self._PKPhi @ scipy.linalg.solve(
                Phi.T @ Phi + np.eye(self._nM) * self.regularizer,
                Phi.T @ Y,
                assume_a="pos",
            )
        elif self.solver == "RKHS-QR":
            A = np.vstack(
                [KNM @ self._PKPhi, np.sqrt(self.regularizer) * np.eye(self._nM)]
            )
            Q, R = np.linalg.qr(A)
            self._weights = self._PKPhi @ scipy.linalg.solve_triangular(
                R, Q.T @ np.vstack([Y, np.zeros((self._nM, Y.shape[1]))])
            )
        elif self.solver == "QR":
            A = np.vstack([KNM, self._VMM])
            Q, R = np.linalg.qr(A)
            self._weights = scipy.linalg.solve_triangular(
                R, Q.T @ np.vstack([Y, np.zeros((KNM.shape[1], Y.shape[1]))])
            )
        elif self.solver == "Normal":
            self._weights = scipy.linalg.solve(
                KNM.T @ KNM
                + self.regularizer * self.KMM
                + np.eye(self._nM) * self.jitter * self._jitter_scale,
                KNM.T @ Y,
                assume_a="pos",
            )

    def predict(self, KTM):
        return KTM @ self._weights

class SparseKernelRidge(BaseEstimator, SparseGPRSolver):

    pass

class SASparseKernelRidge(SparseKernelRidge):
    """ Symmetry-adapted kernel ridge regression class """

    def __init__(self, L, active, active0, zeta=[1], zeta_w=None, cv=2, alpha=1, alphas=None,
                 fit_intercept=False, jitter=1e-15, scoring='neg_root_mean_squared_error'):
        self.L = L
        # L>0 components have zero mean by symmetry
        if L>0:
            fit_intercept = False
        self.cv = SASplitter(L, cv)
        if not hasattr(zeta,'__len__'):
            zeta = [zeta]
        if zeta_w is None:
            zeta_w = np.ones(len(zeta))
        self.zeta = zeta
        self.zeta_w = zeta_w
        self.alpha=alpha
        self.alphas = alphas
        self.scoring = scoring
        self.fit_intercept = fit_intercept

        self.nactive = len(active)
        self.active0 = active0.reshape(active0.shape[:2])
        self.active = np.moveaxis(active, -1,1).reshape((-1,active.shape[1]))
        
        print("Sparse GPR, for nactive= ", self.nactive)
        start = time()
        K0 = self.active0@self.active0.T
        KLMM = (self.active@self.active.T).reshape((self.nactive,2*self.L+1,
                                            self.nactive,2*self.L+1))
        #K0 = np.einsum("iq,jq->ij", self.active0[...,0], self.active0[...,0])
        #KLMM = np.einsum("iqm,jqn->imjn", self.active, self.active)
        self.KLscale = np.trace(KLMM.reshape( ((2*self.L+1)*self.nactive,(2*self.L+1)*self.nactive) ))/self.nactive
        self.K0scale = np.trace(K0)/self.nactive
        if self.K0scale == 0.0 :  # handles gently 0-valued features
            self.K0scale = 1
        if self.KLscale == 0.0 :
            self.KLscale = 1
            
        Kpoly = KLMM*0.0
        for z, zw in zip(self.zeta, self.zeta_w):
            Kpoly += np.einsum("imjn,ij->imjn",KLMM/self.KLscale, zw*(K0/self.K0scale)**(z-1))

        print("KMM compute ", time()-start)
        start = time()
        self.KLMM_flat = Kpoly.reshape( ((2*self.L+1)*self.nactive,((2*self.L+1)*self.nactive)) )
        super(SASparseKernelRidge, self).__init__(KMM = self.KLMM_flat,
                                                  regularizer=alpha, jitter=jitter, solver="RKHS")
        print("KMM init ", time()-start)

    def fit(self, Xm, Ym, X0):
        # this expects properties in the form [i, m] and features in the form [i, q, m]
        # in order to train a SA-GPR model the m indices have to be moved and merged with the i
        # it also gets *invariant* features to build non-linear sa-gpr kernels

        print("Fitting, ", Xm.shape)
        # computes lambda-soap kernel
        start = time()
        X0_flat = X0.reshape(X0.shape[:2])
        Xm_flat = np.moveaxis(Xm, -1,1).reshape((-1,Xm.shape[1]))
        K0NM = X0_flat@self.active0.T
        KLNM = (Xm_flat@self.active.T).reshape((Xm.shape[0],Xm.shape[-1],self.nactive,Xm.shape[-1]))

        #K0NM = np.einsum("iq,jq->ij", X0[...,0], self.active0[...,0])
        #KLNM = np.einsum("iqm,jqn->imjn", Xm, self.active)
        Kpoly = KLNM*0.0
        for z, zw in zip(self.zeta, self.zeta_w):
            Kpoly += np.einsum("imjn,ij->imjn",KLNM/self.KLscale, zw*(K0NM/self.K0scale)**(z-1))
        KLNM_flat = Kpoly.reshape((KLNM.shape[0]*KLNM.shape[1],KLNM.shape[2]*KLNM.shape[3]))
        Ym_flat = Ym.flatten()
        self._Y_mean = 0
        if self.fit_intercept:
            self._Y_mean = Ym_flat.mean()
        Ym_flat = Ym_flat - self._Y_mean
        print("KNM compute", time()-start)

        start = time()
        if self.alphas is not None:
            # determines alpha by grid search
            krcv = SparseKernelRidge(KMM = self.KLMM_flat, jitter=self.jitter)#, rkhs_vectors=self.rkhs_vectors, )
            gscv = GridSearchCV(krcv, dict(regularizer=self.alphas), cv=self.cv, scoring=self.scoring)
            gscv.fit(KLNM_flat, Ym_flat)
            self.cv_stats = gscv.cv_results_
            self.regularizer = gscv.best_params_["regularizer"]
        print("CV ", time()-start)
        self.alpha = self.regularizer # offer common interface - should really rename regularizer upstream
        start = time()
        super(SASparseKernelRidge, self).fit(KLNM_flat, Ym_flat)
        print("KNM fit", time()-start)

    def predict(self, Xm, X0):


        X0_flat = X0.reshape(X0.shape[:2])
        Xm_flat = np.moveaxis(Xm, -1,1).reshape((-1,Xm.shape[1]))
        K0NM = X0_flat@self.active0.T
        KLNM = (Xm_flat@self.active.T).reshape((Xm.shape[0],Xm.shape[-1],self.nactive,Xm.shape[-1]))

        #K0NM = np.einsum("iq,jq->ij", X0[...,0], self.active0[...,0])
        #KLNM = np.einsum("iqm,jqn->imjn", Xm, self.active)
        Kpoly = KLNM*0.0
        for z, zw in zip(self.zeta, self.zeta_w):
            Kpoly += np.einsum("imjn,ij->imjn",KLNM/self.KLscale, zw*(K0NM/self.K0scale)**(z-1))
        Y = self._Y_mean + super(SASparseKernelRidge, self).predict(Kpoly.reshape((KLNM.shape[0]*KLNM.shape[1],KLNM.shape[2]*KLNM.shape[3])))
        return Y.reshape((-1, 2*self.L+1))

class FockRegression:
    """ Collects all the models that are needed to fit and predict a Fock matrix in coupled-momentum blocks form. """

    def __init__(self, orbs, *args, **kwargs):
        # these are the arguments to Ridge - we just store them here because ATM
        # we don't know what we'll be learning
        self._orbs = orbs
        _, self._eldict = orbs_base(orbs)
        self._args = args
        self._kwargs = kwargs
        # guess what we want to do by the arguments
        if "active" in self._kwargs:
            self._model_template= SASparseKernelRidge
        elif "zeta" in self._kwargs:
            self._model_template= SAKernelRidge
        else:
            self._model_template = SARidge
        if "fit_intercept" in self._kwargs:
            self.fit_intercept = self._kwargs["fit_intercept"]
        else:
            self.fit_intercept = "auto"
            
        if "jitter" in self._kwargs:
            self.jitter = self._kwargs["jitter"]

        if "active" in self._kwargs:
            self.active = self._kwargs["active"]
            self.active0 = self._kwargs["active0"]

    def fit(self, feats, fock_bc, slices=None, progress=None):
        self._models = {}
        self.cv_stats_ = {}
        for k in fock_bc.keys():
            self._models[k] = {}
            pkeys = fock_bc[k].keys()
            if progress is not None:
                pkeys = progress(fock_bc[k].keys())
            for orb in pkeys:
                try:   # fancier info if this is tqdm
                    pkeys.set_description("Fitting % 5s: % 20s" % (k, str(orb)))
                except:
                    print("Fitting % 5s: % 20s" % (k, str(orb)))
                    pass
                self._models[k][orb] = {}
                el = self._eldict[(orb[0], orb[1])]
                elb = self._eldict[(orb[2], orb[3])]
                pil1l2 = (1-2*(orb[1]%2))*(1-2*(orb[3]%2)) # parity pi(l1) * pi(l2)
                if slices is None:
                    sl = slice(None)
                else:
                    sl = slices[k][orb]
                for L in fock_bc[k][orb]:
                    # override fit_intercept depending on parameters
                    self._kwargs["fit_intercept"] = self.fit_intercept
                    if self.fit_intercept == "auto":
                        self._kwargs["fit_intercept"] = (L==0 and k == "diag")
                    pi = pil1l2*(1-2*(L%2))
                    if (el,L,pi) in feats[k]:
                        fblock = (el, L, pi)
                    else:
                        fblock = (el, elb, L, pi)
                    if "active" in self._kwargs:
                        self._kwargs["active"] = self.active[k][fblock]
                        self._kwargs["active0"] = self.active0[k][fblock[:-2]+(0, 1)]
                    tgt = fock_bc[k][orb][L][sl]
                    if "jitter" in self._kwargs:
                        self._kwargs["jitter"] = self.jitter
                    try:
                        self._models[k][orb][L] = self._model_template(L, *self._args, **self._kwargs)
                        # determines parity of the block                    
                        self._models[k][orb][L].fit(feats[k][fblock][sl], tgt, X0=feats[k][fblock[:-2]+(0, 1)][sl])
                    except np.linalg.LinAlgError:
                        # handles with error in solve due to small jitter
                        print("solve failure, retrying with larger jitter")
                        self._kwargs["jitter"] *= 100
                        self._models[k][orb][L] = self._model_template(L, *self._args, **self._kwargs)
                        self._models[k][orb][L].fit(feats[k][fblock][sl], tgt, X0=feats[k][fblock[:-2]+(0, 1)][sl])                        
                    self.cv_stats_[(k, orb,L)] = self._models[k][orb][L].cv_stats


    def predict(self, feats, progress=None):
        fock_bc = {}
        for k in self._models.keys():
            fock_bc[k] = {}
            pkeys = self._models[k].keys()
            if progress is not None:
                pkeys = progress(pkeys)
            for orb in pkeys:
                try:   # fancier info if this is tqdm
                    pkeys.set_description("Predicting % 5s: % 20s" % (k, str(orb)))
                except:
                    pass
                fock_bc[k][orb] = {}
                el = self._eldict[(orb[0], orb[1])]
                elb = self._eldict[(orb[2], orb[3])]
                pil1l2 = (1-2*(orb[1]%2))*(1-2*(orb[3]%2)) # parity pi(l1) * pi(l2)

                #if progress is not None:
                #    print("Orbital: ", el, elb, orb)

                for L in self._models[k][orb]:
                    pi = pil1l2*(1-2*(L%2))
                    if (el,L,pi) in feats[k]:
                        if len(feats[k][(el,L,pi)]) > 0:
                            fock_bc[k][orb][L] = self._models[k][orb][L].predict(feats[k][(el,L,pi)], X0=feats[k][(el,0,1)])
                    else:
                        if len(feats[k][(el,elb,L,pi)]) > 0:
                            fock_bc[k][orb][L] = self._models[k][orb][L].predict(feats[k][(el,elb,L,pi)], X0=feats[k][(el,elb,0,1)])
        return fock_bc

def active_set_selection(feats, blocks, orbs, selector, normalize=True, slices=None):
    """ Compute active set points for the blocks given. """

    n_to_select = selector.n_to_select  # makes sure the selector has the "right" interface
    # determines active set 
    active_feats0 = {}
    active_feats = {}
    active_idx = {}
    for tblock in blocks.keys():
        active_idx[tblock] = {}
        active_feats[tblock] = {}
        active_feats0[tblock] = {}
        for kblock in blocks[tblock]:
            # gets the feature block corresponding to the invariant features (l=0, sigma=1)
            fblock0 = block_to_feat_index(tblock, kblock, 0, orbs)[:-1]+(1,)
            if slices is None: # if we must only consider a training slice...
                islice = slice(None)
            else:
                islice = slices[tblock][kblock]
                
            if fblock0 in active_idx[tblock]:  # reuse indices when available
                sel_idx = active_idx[tblock][fblock0]
            else:
                xblock0 = feats[tblock][fblock0][islice]
                xblock0 = xblock0.reshape(len(xblock0),-1)
                if normalize:
                    mean_sz = np.mean(((xblock0-xblock0.mean(axis=0))**2).sum(axis=1))
                    if mean_sz > 0:
                        xblock0 = xblock0/mean_sz
                selector.n_to_select = min(xblock0.shape[0]-1, n_to_select)
                try:
                    selector.fit(xblock0)
                except np.linalg.LinAlgError:
                    print(f"Error during selection. Stopping at {len(selector.n_selected_)}/{selector.n_to_select} active points.")
                except error:
                    print(f"Uncaught error for {fblock0}, selected {selector.n_selected_}")
                    raise error                
                print(f"Selected {selector.n_selected_} for {kblock} [{fblock0}]")
                sel_idx = selector.selected_idx_[:selector.n_selected_]
                selector.n_to_select = n_to_select
            
            active_idx[tblock][fblock0] = sel_idx
            active_feats0[tblock][fblock0] = feats[tblock][fblock0][islice][sel_idx]
            for l in blocks[tblock][kblock]: # these will only generate slices so there's no harm in being redundant
                fblock = block_to_feat_index(tblock, kblock, l, orbs)
                active_feats[tblock][fblock] = feats[tblock][fblock][islice][sel_idx]
                
    return active_idx, active_feats0, active_feats
