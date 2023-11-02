import numpy as np
from numpy.fft import fft, ifft
from numpy.linalg import cholesky
from scipy.linalg import block_diag
from scipy.linalg import lstsq, svd
from scipy.linalg import norm
from scipy.sparse import eye
from scipy.sparse import spdiags
from scipy.signal import convolve2d
from scipy.stats import shapiro
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sympy
from sympy import symbols, diff, lambdify, sympify
import utils

class WENDy:
    """
    Inputs:
       mt_min: min radius
       mt_max: max radius
       K_min : min number of test fuctions
       K_max : max number of test fuctions
       phifun: test function
       mt_params: tf radii 
       toggle_VVp_svd # 0, no SVD reduction; in (0,1), truncates Frobenious norm; NaN, truncates SVD according to cornerpoint of cumulative sum of singular values
       w0 : first guess
       use_true: use true soln

    """
    def __init__(self, mt_min = None, mt_max = None,  K_max = 5000, K_min = None, phifun = lambda x: np.exp(-9*(1-x**2)**(-1)), mt_params = [2**i for i in range(4)], toggle_VVp_svd = 0, w0 = None, use_true=False):
        self.mt_min = mt_min
        self.mt_max = mt_max
        self.K_min = K_min
        self.K_max = K_max
        self.phifun = phifun
        self.meth = 'mtmin'
        self.submt  = 3
        self.mt_params = mt_params
        self.center_scheme = 'uni'
        self.toggle_VVp_svd = toggle_VVp_svd # 0, no SVD reduction; in (0,1), truncates Frobenious norm; NaN, truncates SVD according to cornerpoint of cumulative sum of singular values
        self.w0 = w0
        self.use_true=use_true
        #Jacobian correction params
        self.err_norm = 2
        self.iter_diff_tol = 1e-6
        self.max_iter = 100
        self.diag_reg = 1e-4
        self.pvalmin = 1e-4
        self.check_pval_it = 10

        


    def fit(self, xobs, tobs, features, true_vec = None):
        self.features = features
        self.xobs = xobs
        self.tobs = tobs
        self.true_vec = true_vec
        
        # get dimensions
        param_length_vec = np.array([len(x) for x in features])
        eq_inds = np.array([len(x) > 0 for x in features])
        num_eq = np.sum(eq_inds)
        M, nstates = xobs.shape
        self.M = M
        self.nstates = nstates

        #set parameters:
        if self.K_min is None: 
            self.K_min = len(features) * 2
        if self.mt_max is None: 
            self.mt_max = max((M-1)//2 - self.K_min, 1)
        if self.mt_min is None:
            self.mt_min = self.rad_select(tobs, xobs, self.phifun, 1, self.submt, 0, 1, 2, self.mt_max, None)
        
        self.mt_cell = [[self.phifun, self.meth, x] for x in self.mt_params]


        # estimate noise variance and build initial covariance
        sig_ests = np.array([self.estimate_sigma(xobs[:, i]) for i in range(nstates)])
        RT_0 = spdiags(np.kron(sig_ests, np.ones(M)), 0, M*nstates, M*nstates)

        # get test function
        cm = len(self.mt_cell)
        cn = 1
        K = min(int(self.K_max / nstates / cm), M)

        if cn < nstates:
            mt = np.zeros((cm, nstates))
            #print(mt.shape)
            for i in range(cm):
                mt_cell_val = self.mt_cell[i]
                for j in range(nstates): 
                    mt_temp = self.get_rad(xobs[:, j], tobs, mt_cell_val[0], mt_cell_val[1], mt_cell_val[2], self.mt_min, self.mt_max)
                    mt[i, j] = mt_temp
            mt = np.ceil(1. / np.mean(1. / mt, axis=1))
            if self.toggle_VVp_svd != 0 and cm > 1:
                V = np.concatenate([self.get_VVp_svd(int(y), tobs, K, x[0], self.center_scheme) for x, y in zip(self.mt_cell, mt)])
                V, Vp = self.VVp_svd(V, self.K_min, tobs, self.toggle_VVp_svd)
            else: 
                V, Vp = zip(*[self.get_VVp(int(y), tobs, 1, K, x[0], self.center_scheme) for x, y in zip(self.mt_cell, mt)])
                V, Vp = np.concatenate(V), np.concatenate(Vp)
            V_cell = [V] * nstates
            Vp_cell = [Vp] * nstates
            mt = np.tile(mt, (1, nstates))    
        else:
            mt = np.zeros((cm, nstates))
            for i in range(cm):
                mt_cell_val = self.mt_cell[i]
                for j in range(nstates): 
                    mt_temp = self.get_rad(xobs[:, j], tobs, mt_cell_val[0], mt_cell_val[1], mt_cell_val[2], self.mt_min, self.mt_max)
                    mt[i, j] = mt_temp
            mt = np.ceil(1. / np.mean(1. / mt, axis=1)).reshape(1, -1).T
            V_cell = [None]*num_eq
            Vp_cell = [None]*num_eq
            for nn in range(num_eq):
                V_cell_temp  = np.empty_like(tobs).reshape(1, -1)
                Vp_cell_temp = np.empty_like(tobs).reshape(1, -1)
                for j in range(np.count_nonzero(mt[:, nn])):
                    #print(mt[j,nn])
                    if mt[j,nn]:
                        if self.toggle_VVp_svd != 0 and cm > 1:
                            V_cell_temp = np.vstack((V_cell_temp, self.get_VVp_svd(int(mt[j,nn]), tobs, K, self.mt_cell[min(nn,cn-1)][0], self.center_scheme)))
                        else:
                            V, Vp = self.get_VVp(int(mt[j,nn]), tobs, 1, K, self.mt_cell[min(nn,cn-1)][0], self.center_scheme)
                            V_cell_temp = np.vstack((V_cell_temp, V))  
                            Vp_cell_temp = np.vstack((Vp_cell_temp, Vp))           
                V_cell_temp = V_cell_temp[1:, :]
                Vp_cell_temp = Vp_cell_temp[1:, :]
                V_cell[nn] = V_cell_temp
                Vp_cell[nn] = Vp_cell_temp
                if self.toggle_VVp_svd != 0 and cm > 1:
                    Vc, Vcp = self.VVp_svd(V_cell[nn], self.K_min,tobs, self.toggle_VVp_svd)
                    V_cell[nn] = Vc
                    Vp_cell[nn] = Vcp
                
        # build linear system
        xobs_cell = [xobs[:,i] for i in range(nstates)]
        Theta_cell = [np.vstack([y(*xobs_cell) for y in x]).T for x in features]
        G_0 = [V.dot(x) for x, V in zip(Theta_cell, V_cell)]
        G_0 = block_diag(*G_0)
        b_0 = np.hstack([-Vp.dot(x) for x, Vp in zip(xobs_cell, Vp_cell)]).reshape(-1,1)

        #build library Jacobian
        Jac_mat = self.build_Jac_sym(features,xobs)
        if self.max_iter > 1: 
            L0, L1 = self.get_Lfac(Jac_mat,param_length_vec,V_cell,Vp_cell)
            L0 = L0@RT_0
            #not sure this would work
            s1, s2, s3 = L1.shape
            L1_temp = L1.copy()
            for i in range(s3):
                L1_temp[:, :, i] = L1[:, :, i]@RT_0 
                L1 = L1_temp

        
        # initialize
        pvals_list = []
        if not self.w0:
            w0 = self.windy_opt(G_0, b_0, meth='LS', batch_size=1, num_runs=1, avg_meth='mean', cov=None).reshape(-1,1)
        w_hat = w0
        w_hat_its = w_hat
        res = G_0 @ w_hat - b_0

        '''''
        #res_true = G_0 @ true_vec - b_0
        #res_0 = res
        #res_0_true = res_true

        if self.err_norm > 0:
            errs = norm(w0 - true_vec, ord=self.err_norm) / norm(true_vec, ord=self.err_norm)
        else:
            errs = norm(abs(w0 - true_vec) / abs(true_vec), ord=-self.err_norm)
        '''''
        iter = 1;check = 1;pval = 1

        RT = eye(len(b_0), format='csc')
        _, pvals = shapiro(res)
        pvals_list.append(pvals)

        while check > self.iter_diff_tol and iter < self.max_iter and pval > self.pvalmin:
            # update covariance
            if self.use_true:
                RT, _, _, _ = self.get_RT(L0, L1, true_vec, self.diag_reg)
            else:
                RT, _, _, _ = self.get_RT(L0, L1, w_hat, self.diag_reg)
        
            G = np.linalg.solve(RT, G_0)
            b = np.linalg.solve(RT, b_0)

            # update parameters
            w_hat = self.windy_opt(G, b, meth='LS', batch_size=1, num_runs=1, avg_meth='mean', cov=None).reshape(-1,1)
            res_n = G.dot(w_hat) - b

            # check stopping conditions
            _, pvals = shapiro(res_n)
            pvals_list.append(pvals)
            if iter+1 > self.check_pval_it:
                pval = pvals_list[iter]
            check = np.linalg.norm((w_hat_its[:, -1].reshape(-1, 1) - w_hat)) / np.linalg.norm(w_hat_its[:, -1].reshape(-1,1))
            iter += 1
            

            # collect quantities of interest
            #res = np.hstack((res, res_n.reshape((-1, 1))))
            #res_true = np.hstack((res_true, (G.dot(true_vec) - b).reshape((-1, 1))))
            #res_0 = np.hstack((res_0, (G_0.dot(w_hat) - b_0).reshape((-1, 1))))
            #res_0_true = np.hstack((res_0_true, (G_0.dot(true_vec) - b_0).reshape((-1, 1))))
            w_hat_its = np.hstack((w_hat_its, w_hat.reshape((-1, 1))))
            #if self.err_norm > 0:
                #errs = np.hstack((errs, np.linalg.norm(w_hat - true_vec, self.err_norm) / np.linalg.norm(true_vec, self.err_norm)))
            #else:
                #errs = np.hstack((errs, np.linalg.norm(np.abs(w_hat - true_vec) / np.abs(true_vec), -self.err_norm)))

        if pval < self.pvalmin:
            print('error: WENDy iterates diverged')
            ind = np.argmax(pvals)
            w_hat = w_hat_its[:, ind]
            #res = np.hstack((res, res[:, ind].reshape((-1, 1))))
            #res_true = np.hstack((res_true, res_true[:, ind].reshape((-1, 1))))
            #res_0 = np.hstack((res_0, res_0[:, ind].reshape((-1, 1))))
            #res_0_true = np.hstack((res_0_true, res_0_true[:, ind].reshape((-1, 1))))
            w_hat_its = np.hstack((w_hat_its, w_hat_its[:, ind].reshape((-1, 1))))
            #errs = np.hstack((errs, errs[ind]))
        #Ginv = lstsq(G_0, RT)[0]
        #CovW = Ginv.dot(Ginv.T)
        #stdW = np.sqrt(np.diag(CovW))
        #mseW = (np.mean(res[:, -1] ** 2))
        self.w_hat  = w_hat
        return w_hat
    
    def simulate(self, x0, t):
        tol_ode = 1e-8
        w_hat_tolist = []
        count = 0
        for i in range(len(self.features)): 
            a = self.features[i]
            coef = []
            for j in range(len(a)):
                coef.append(self.w_hat[count+j][0])
            count = count + len(a)
            w_hat_tolist.append(coef)

        def rhs_fun(features, params, x):
            nstates = len(x)
            x = tuple(x)
            dx = np.zeros(nstates)
            for i in range(nstates):
                dx[i] = np.sum([f(*x)*p for f, p in zip(features[i], params[i])])
            return dx
        rhs_p = lambda t, x: rhs_fun(self.features, w_hat_tolist, x)
        sol = solve_ivp(rhs_p, t_span = np.array([t[0], t[-1]]), y0=x0, t_eval=t, rtol=tol_ode, atol=tol_ode)
        return sol.y.T
    
    def rad_select(self, t0, y, phifun, inc, sub, q, s, m_min, m_max, pow):
        if phifun is None:
            mt = m_min
        else:
            M, nstates = y.shape
            dt = np.mean(np.diff(t0))
            #print("dt", dt)
            t = t0
            if q > 0:
                t_mid = t[t.shape[0] // 2]
                prox_u = lambda t: np.exp(-np.power(np.abs(t-t_mid), q))
                prox_u_vec = (dt / inc / np.sqrt(M*dt)) * np.fft.fftshift(np.fft.fft(prox_u(t)))
            else:
                if inc > 1:
                    y_interp = interp1d(t0, y, kind='spline', axis=0)(t)
                    prox_u_vec = (dt / inc / np.sqrt(M*dt)) * np.fft.fftshift(np.fft.fft(y_interp, axis=0))
                elif inc == 1:
                    prox_u_vec = dt / np.sqrt(M*dt) * np.fft.fftshift(np.fft.fft(y, axis=0))
        
            errs = []
            ms = []      
            for m in range(m_min, m_max+1):
                t_phi = np.linspace(-1+dt/inc, 1-dt/inc, 2*inc*m-1)
                Qs = np.arange(0, len(t)-2*inc*m+1, np.floor(s*inc*m), dtype=int)
                errs_temp = np.zeros((nstates, len(Qs)))
                #print(Qs)
                
                for Q in range(len(Qs)):
                    phi_vec = np.zeros_like(t)
                    phi_vec[Qs[Q]:Qs[Q]+len(t_phi)] = phifun(t_phi)
                    phi_vec = phi_vec/np.linalg.norm(phi_vec)
                    for nn in range(nstates):
                        phiu_fft = (dt/np.sqrt(M*dt)) * np.fft.fft(phi_vec * y[:, nn])  
                        alias = phiu_fft[0:int(inc*M/2):int(M/sub)]
                        errs_temp[nn, Q] = 2 * (2 * np.pi / np.sqrt(M*dt)) * np.sum(np.arange(alias.shape[0]) * np.imag(alias), axis=0)
                check1 = np.sqrt(np.mean(np.power(errs_temp, 2)))
                errs.append(check1)
                ms.append(m)
            
            log_errs = np.log(errs)
            if isinstance(pow, str):
                #b, _ = findchangepts(-log_errs, prominence=1, distance=1)
                if len(b) == 0:
                    b = np.argmin(np.array(errs) * np.power(np.array(ms), 0.5))
                else:
                    b = b[0]
            elif pow is None:
                b = self.getcorner(log_errs, ms)
            else:
                b = np.argmin(np.array(errs) * np.power(np.array(ms), pow))
            mt = ms[b]
        return mt

    def get_RT(self, L0, L1, w, diag_reg): 
        dims = L1.shape
        if not np.all(np.all(w == 0)):
            L0 = L0 + np.reshape(np.transpose(L1, (2, 0, 1)).reshape(dims[2], -1).T @ w, (dims[0], -1))
        Cov = L0 @ L0.T
        RT = cholesky((1-diag_reg)*Cov + diag_reg*np.diag(np.diag(Cov)))
        return RT, L0, Cov, diag_reg

    def windy_opt(self, G, b, meth='LS', batch_size=1, num_runs=1, avg_meth='mean', cov=None):
        if meth == 'LS':
            if cov is None:
                w = lstsq(G, b)[0]
            else:
                w = lstsq(G, b, rcond=None, overwrite_a=False, overwrite_b=False, check_finite=True, lapack_driver=None, **{'cov': cov})[0]
        elif meth == 'TLS':  #never test
            _, _, V = svd(np.hstack((G, b)), full_matrices=False)
            n = G.shape[1] 
            w = V[:n, n:].T
        elif meth == 'ensLS':  #never test
            w = els(G, b, batch_size, num_runs, avg_meth)
        else:
            raise ValueError(f"Unknown method: {meth}")
        return w

    def get_Lfac(self, Jac_mat, Js, V_cell, Vp_cell):
        _, d, M = Jac_mat.shape
        Jac_mat = np.transpose(Jac_mat, (1, 2, 0))
        eq_inds = np.where(Js)[0]
        num_eq = len(eq_inds)
        L0 = block_diag(*Vp_cell)
        L1 = np.zeros((L0.shape[0], d*M, sum(Js)))
        Ktot = 0
        Jtot = 0
        for i in range(num_eq):
            K, _ = V_cell[i].shape
            J = Js[eq_inds[i]]
            for ell in range(d):
                m = np.expand_dims(Jac_mat[ell, :, Jtot+(np.arange(J))].T, axis = 0)
                n = V_cell[i][:, :, np.newaxis]
                ixgrid = np.ix_(range(Ktot, Ktot + K), range(ell*M, (ell+1)*M), range(Jtot, Jtot + J))
                L1[ixgrid] = m*n
            Ktot = Ktot + K
            Jtot = Jtot + J
        return L0, L1

    def build_Jac_sym(self, features, xobs):
        M, nstates = xobs.shape
        features = [f for f_list in features for f in f_list]
        J = len(features)
        Jac_mat = np.zeros((J, nstates, M))

        # Create the symbolic variables
        args = symbols('x0:%d' % nstates)
    
        def diff_lambda(f, var):
            #args = symbols('x0:%d' % f.__code__.co_argcount)
            return sympify(diff(f(*args), var))

        for j in range(J):
            f = features[j]
            for state in range(nstates): 
                g = diff_lambda(f, args[state])
                G = lambdify(args, g, 'numpy')
                for i in range(M):
                    x_val = xobs[i, :]
                    z = G(*x_val)
                    Jac_mat[j, state , i] =  z
        return Jac_mat

    def get_VVp(self, mt, t, max_d, K, phifun=None, center_scheme='uni'):
        dt = np.mean(np.diff(t))
        M = len(t)
        
        if phifun is not None:
            Cfs = self.phi_weights(phifun, mt, max_d)
        else:
            Cfs = np.vstack([[0, 0], self.fdcoeffF(1, t[mt], t[0:mt*2-1])])    #not tested here
            Cfs[1, :] *= -(mt*dt)
        v = Cfs[-2, :].dot((mt*dt)**(-max_d+1)) * dt
        vp = Cfs[-1, :].dot((mt*dt)**(-max_d)) * dt
        
        if center_scheme == 'uni':
            gap = max(1, np.floor((M-2*mt)/K).astype(int))
            diags = np.arange(0, M-2*mt, gap, dtype=int)
            diags = diags[:min(K, len(diags))]
            V = np.zeros((len(diags), M))
            Vp = np.zeros((len(diags), M))
            for j in range(len(diags)):
                V[j, gap*(j):gap*(j)+2*mt+1] = v
                Vp[j, gap*(j):gap*(j)+2*mt+1] = vp
        elif center_scheme == 'random':  #also not tested here
            gaps = np.random.permutation(M-2*mt)[:K]
            V = np.zeros((K, M))
            Vp = np.zeros((K, M))
            for j in range(K):
                V[j, gaps[j]:gaps[j]+2*mt+1] = v
                Vp[j, gaps[j]:gaps[j]+2*mt+1] = vp  
        elif isinstance(center_scheme, np.ndarray):  #not tested
            center_scheme = np.unique(np.maximum(np.minimum(center_scheme, M-mt), mt+1))
            K = len(center_scheme)
            V = np.zeros((K, M))
            Vp = np.zeros((K, M))
            for j in range(K):
                V[j, center_scheme[j]-mt:center_scheme[j]+mt+1] = v
                Vp[j, center_scheme[j]-mt:center_scheme[j]+mt+1] = vp     
        return V, Vp

    def VVp_svd(self, V, K_min, t, toggle_VVp_svd):
        m = len(t)
        dt = np.mean(np.diff(t))
        U, S, _  = svd(V.T, full_matrices=False)
        sings = np.diag(S)

        def getcorner(Ufft, xx):
            NN = len(Ufft)
            Ufft = Ufft / max(np.abs(Ufft)) * NN
            errs = np.zeros(NN)
            for k in range(1, NN+1):
                L1, L2, m1, m2, b1, b2, Ufft_av1, Ufft_av2 = self.build_lines(Ufft, xx, k)
                #errs[k-1] = np.sqrt(np.sum(((L1-Ufft_av1) / Ufft_av1)**2) + np.sum(((L2-Ufft_av2) / Ufft_av2)**2)) # relative l2
                errs[k-1] = (np.sum(np.abs((L1-Ufft_av1) / Ufft_av1)) + np.sum(np.abs((L2-Ufft_av2) / Ufft_av2))) # relative l1
            #print("get_corner err", errs)
            tstarind = np.nanargmin(errs)
            return tstarind 

        if toggle_VVp_svd > 0:
            s = np.argmax(np.cumsum(sings**2)/np.sum(sings**2) > toggle_VVp_svd**2)
            if s == 0:
                s = min(K, V.shape[0])
        else:
            corner_data = np.cumsum(S)/np.sum(S)
            s = getcorner(corner_data, np.arange(0, len(corner_data))) +1
            s = min(max(K_min, s), V.shape[0])

        inds = np.arange(s)
        V = U[:, inds].T*dt
        Vp = V.T
        Vp_hat = fft(Vp, axis = 0)
        k = (2*np.pi/(m*dt))*np.arange(-m/2, m/2)
        k = np.fft.fftshift(k)
        Vp_hat =  Vp_hat*k.reshape(-1, 1)*(1j)
        Vp = np.real(ifft(Vp_hat, axis = 0)).T
        return V, Vp

    def phi_weights(self, phifun, m, maxd):
        xf = np.linspace(-1, 1, 2*m+1)
        x = xf[1:-1]
        Cfs = np.zeros((maxd+1, 2*m+1))
        y = sympy.symbols('y')
        #f = lambda y: phifun(y)
        eta = 9
        f = lambda y: sympy.exp(-eta*(1-y**2)**(-1))
        for j in range(1, maxd+2):
            Df = sympy.lambdify(y, sympy.diff(f(y), y, j-1))
            Cfs[j-1, 1:-1] = np.nan_to_num(Df(x), nan=Df(np.finfo(float).eps))
            inds = np.where(np.isinf(np.abs(Cfs[j-1,:])))[0]
            for k in range(len(inds)):
                Cfs[j-1, inds[k]] = Df(xf[inds[k]] - np.sign(xf[inds[k]])*np.finfo(float).eps)
        Cfs = Cfs / np.linalg.norm(Cfs[0,:], 2)
        return Cfs
    
    def get_VVp_svd(self, mt, t, K, phifun, center_scheme):
        dt = np.mean(np.diff(t))
        M = len(t)
        Cfs = self.phi_weights(phifun, mt, 1)
        v = Cfs[0,:] * dt
        if center_scheme == 'uni':
            gap = max(1, int(np.floor((M - 2*mt) / K)))
            diags = np.arange(0, M-2*mt, gap)
            diags = diags[:min(K, len(diags))]
            V = np.zeros((len(diags), M))
            for j in range(len(diags)):
                V[j, gap*(j):gap*(j)+2*mt+1] = v
        elif center_scheme == 'random':
            gaps = np.random.permutation(np.arange(M-2*mt))
            gaps = gaps[:K]
            V = np.zeros((K, M))
            for j in range(K):
                V[j, gaps[j] : gaps[j]+2*mt+1] = v
        elif isinstance(center_scheme, float):
            center_scheme = np.unique(np.maximum(np.minimum(center_scheme, M-mt), mt+1))
            K = len(center_scheme)
            V = np.zeros((K, M))
            for j in range(K):
                V[j, int(center_scheme[j]-mt) : int(center_scheme[j]+mt+1)] = v
        return V
    
    def get_rad(self, xobs, tobs, phifun, meth, p, mt_min, mt_max):
        if (phifun is None) or (meth is None) or (p is None):
            mt = 0
        else:
            if meth == 'direct':
                mt = p
            elif meth == 'FFT':
                _, _, _, mt = findcorners(xobs, tobs, [], p, phifun)
            elif meth == 'timefrac':
                mt = int(len(tobs) * p)
            elif meth == 'mtmin':
                mt = p * mt_min
            mt = min(mt, mt_max)
        return mt
   
    def fdcoeffF(self, k, xbar, x):
        n = len(x)
        if k >= n:
            raise ValueError('*** length(x) must be larger than k')

        m = k-1  # change to m=n-1 if you want to compute coefficients for all
                # possible derivatives.  Then modify to output all of C.
        c1 = 1
        c4 = x[0] - xbar
        C = np.zeros((n, m+2))
        C[0, 0] = 1

        for i in range(n-1):
            i1 = i+1
            mn = min(i,m)
            c2 = 1
            c5 = c4
            c4 = x[i1] - xbar      
            for j in range(-1, i):
                j1 = j+1
                c3 = x[i1] - x[j1]
                c2 = c2*c3
                if j == i-1:
                    for s in range(mn, -1, -1):
                        s1 = s+1
                        C[i1, s1] = c1*((s+1)*C[i1-1, s1-1] - c5*C[i1-1, s1])/c2
                    C[i1, 0] = -c1*c5*C[i1-1, 0]/c2
                for s in range(mn, -1, -1):
                    s1 = s+1
                    C[j1, s1] = (c4*C[j1, s1] - (s+1)*C[j1, s1-1])/c3
                C[j1, 0] = c4*C[j1, 0]/c3
            c1 = c2
        return C

    def estimate_sigma(self,f):
        k = 6
        C = self.fdcoeffF(k, 0, np.arange(-k-2, k+3))
        filter = C[:, -1]
        filter = filter / np.linalg.norm(filter, ord=2)
        filter = filter.reshape(1, -1).T
        f = f.reshape(-1, 1)
        sig = np.sqrt(np.mean(np.square(convolve2d(f, filter, mode='valid'))))
        return sig

    def getcorner(self, Ufft, xx):
        NN = len(Ufft)
        Ufft = Ufft / max(np.abs(Ufft)) * NN
        errs = np.zeros(NN)
        for k in range(1, NN+1):
            L1, L2, m1, m2, b1, b2, Ufft_av1, Ufft_av2 = self.build_lines(Ufft, xx, k)
            errs[k-1] = np.sqrt(np.sum(((L1-Ufft_av1) / Ufft_av1)**2) + np.sum(((L2-Ufft_av2) / Ufft_av2)**2)) # relative l2
            #errs[k-1] = (np.sum(np.abs((L1-Ufft_av1) / Ufft_av1)) + np.sum(np.abs((L2-Ufft_av2) / Ufft_av2))) # relative l1
        tstarind = np.nanargmin(errs)
        return tstarind 

    def build_lines(self, Ufft, xx, k):
        NN = len(Ufft)
        subinds1 = np.arange(0, k)
        subinds2 = np.arange(k-1, NN)
        Ufft_av1 = Ufft[subinds1]
        Ufft_av2 = Ufft[subinds2]
        xx = np.array(xx)
        m1, b1, L1 = self.lin_regress(Ufft_av1, xx[subinds1])
        m2, b2, L2 = self.lin_regress(Ufft_av2, xx[subinds2])
        return L1, L2, m1, m2, b1, b2, Ufft_av1, Ufft_av2

    def lin_regress(self, U, x):
        m = (U[-1] - U[0]) / (x[-1] - x[0])
        b = U[0] - m * x[0]
        L = U[0] + m * (x - x[0])
        return m, b, L
