#Copyright 2025, All Rights Reserved.
#Code by April Tran, adapted from original code by Dan Messenger.

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
import scipy
import sympy
from sympy import symbols, diff, lambdify, sympify
import matplotlib.pyplot as plt
import math
from scipy.special import bernoulli
#from joblib import Parallel, delayed
import time
import kneed 

class PyWENDy:
    def __init__(self, features):
        """
        Inputs:
        features: Rhs features
        """
        self.features = features
      
    def fit_IRLS(self, xobs, tobs, radius, type_rad = 0, type_tf = 0, toggle_SVD = False, gap = 1, p = 10, S = 1, mu = [1, 2, 1], Mtilde = None, diag_reg = 1e-10, trunc = 0):
        """
        Inputs:
        xobs: data
        tobs: time
        radius: grid point radius in a list (if None - compute this automatically)
        type_rad: 0 - Single-scale Local, 1 - Multi-scale Global
        type_tf: Type of test function: 0 -> L2, 1 ->L_inf
        toggle_SVD: False -> no SVD,  True -> SVD
        gap: gap between test functions
        p: order of poly tf
        S: truncation order of In
        mu: finite difference orders of accuracy

        trunc: truncation method for svd 0 -> corner point, 0 < trunc < 1 trunc% weight of singularvals
        """

        self.xobs = xobs
        self.tobs = tobs
        self.radius = radius
        self.type_rad = type_rad
        self.type_tf = type_tf
        self.toggle_SVD = toggle_SVD
        self.gap = gap
        self.p = p

        self.d = self.xobs.shape[1]
        self.dt = self.tobs[1]
        self.T = self.tobs[-1]
        self.M = len(self.tobs) - 1
        self.Mp1 = len(self.tobs)

        self.S = S
        self.mu = mu 
        if Mtilde == None: 
            Mtilde = self.M
        self.Mtilde = Mtilde
        self.trunc = trunc

        #set params
        self.iter_diff_tol = 1e-6
        self.max_iter = 100
        self.diag_reg = diag_reg
        self.pvalmin = 1e-4
        self.check_pval_it = 10
        self.tau = 1e-5


        sum2p = 0
        for k in range(2*self.p + 1):
            sum2p = sum2p + math.comb(2*self.p, k)*(-1)**k/(2*k + 1)
        self.sum2p = sum2p

        #build Theta
        xobs_cell = [self.xobs[:,i] for i in range(self.d)]
        Theta_cell = [np.vstack([y(*xobs_cell) for y in x]).T for x in self.features]

        #compute radii
        if self.radius == None: 
            if self.type_rad == 0: #Singlescale Local
                radius = [self.get_r_c_hat()]
            elif self.type_rad == 1: #Multiscale Global
                rad_min = self.get_r_min()
                radius = np.minimum(rad_min * 2**np.arange(4), self.M//2 - 1)
        self.radius = radius

        #build test function matrices
        if len(self.radius) == 1:
            Phi, PhiP, centers = self.getVVp_L2(radius[0])   
        else: 
            Phi = []
            PhiP = []
            centers = []
            for rad in radius: 
                Phi_r, PhiP_r, centers_r = self.getVVp_L2(rad)  
                Phi.append(Phi_r)
                PhiP.append(PhiP_r)
                centers.append(centers_r)
            Phi = np.vstack(Phi)
            PhiP = np.vstack(PhiP)

        #print("Phi shape", Phi.shape)
        if self.toggle_SVD == True: 
            Phi, PhiP = self.svdPhi(Phi, PhiP)
            #print("Phi shape after SVD", Phi.shape)
        self.Phi = Phi
        self.PhiP = PhiP
        self.centers = centers

        
        Jac = self.build_Jac_sym(self.features, self.xobs)
        V_cell = [self.Phi]*self.d
        Vp_cell = [self.PhiP]*self.d
        param_length_vec = np.array([len(x) for x in self.features])
        L0, L1 = self.get_Lfac(Jac,param_length_vec,V_cell,Vp_cell)

        # estimate noise variance and build initial covariance
        sig_ests = np.array([self.estimate_sigma(self.xobs[:, i]) for i in range(self.d)])
        RT_0 = spdiags(np.kron(sig_ests, np.ones(self.Mp1)), 0, self.Mp1*self.d, self.Mp1*self.d)
        L0 = L0 @ RT_0
    
        s1, s2, s3 = L1.shape
        L1_temp = L1.copy()
        for i in range(s3):
            L1_temp[:, :, i] = L1[:, :, i] @ RT_0 
            L1 = L1_temp
        

        # build linear system
        G0 = [Phi @ x for x, Phi in zip(Theta_cell, V_cell)]
        G0 = scipy.linalg.block_diag(*G0)
        b0 = np.hstack([-PhiP @ x for x, PhiP in zip(xobs_cell, Vp_cell)]).reshape(-1,1)
        w0 =  np.linalg.lstsq(G0, b0, rcond=None)[0].reshape(-1, 1)

        pvals_list = []
        res = G0 @ w0 - b0
        _, pvals = scipy.stats.shapiro(res)
        pvals_list.append(pvals)

        w_hat = w0.reshape(-1,1)
        w_hat_its =[w_hat]
        iter = 1;check = 1;pval = 1
        RT = scipy.sparse.eye(len(b0), format='csc')

        while check > self.iter_diff_tol and iter < self.max_iter and pval > self.pvalmin:
            #print(iter)
            try:
                RT, _, _, _ = self.get_RT(L0, L1, w_hat, self.diag_reg)
            except np.linalg.LinAlgError:
                print("Cholesky decomposition failed: matrix not positive definite.")
                print("Returning initial guess w0.")
                self.w_hat = w0
                self.pvals_list = pvals_list
                self.w_hat_its = w_hat_its
                return w0
            G = np.linalg.solve(RT, G0)
            b = np.linalg.solve(RT, b0)

            w_hat = np.linalg.lstsq(G, b, rcond=None)[0].reshape(-1, 1)
            res_n = G @ w_hat  - b
            #plt.hist(res_n)
            #plt.show()

            # check stopping conditions
            _, pvals = scipy.stats.shapiro(res_n)
            pvals_list.append(pvals)
            if iter+1 > self.check_pval_it:
                pval = pvals_list[iter]

            check = np.linalg.norm(w_hat_its[-1] - w_hat)/np.linalg.norm(w_hat_its[-1])
            iter += 1

            w_hat_its.append(w_hat)


        if pval < self.pvalmin:
            print('error: WENDy iterates diverged')
            ind = np.argmax(pvals_list)
            w_hat = w_hat_its[ind]
            w_hat_its.append(w_hat)

        self.w_hat = w_hat
        self.pvals_list = pvals_list
        self.w_hat_its = w_hat_its
        return self.w_hat


    def get_r_min(self):
        radii = np.arange(2, self.M//2, 1) 
        s=3
        ntilde = int(np.floor(self.M/s))

        erms = []
        for radius in radii:
            Phi, _, _ = self.getVVp_L2(radius)   
            K = Phi.shape[0]
            errs_temp = np.zeros((self.d,K))
            for i in range(self.d): #going through each dimension
                for k in range(K):
                    phi_u =  Phi[k]*self.xobs[:, i]
                    phiu_fft = (self.dt/np.sqrt(self.M*self.dt)) * np.fft.fft(phi_u)  
                    errs_temp[i, k] = 2 * (2 * np.pi / np.sqrt(self.M*self.dt)) * ((phiu_fft[ntilde]).imag)
            sum = np.sqrt(np.linalg.norm(errs_temp.flatten(), 2)**2/K)
            erms.append(sum)
        erms = np.array(erms)
        rad = self.getcorner(np.log(erms), radii, l=2)
        return rad


    def svdPhi(self, Phi, PhiP): 
        """ return orthogonal test functions
        """
        u, s, vh = np.linalg.svd(Phi)
        sv_mass = np.cumsum(s)
        if self.trunc == 0:
            r = self.getcorner(sv_mass/np.sum(s), np.arange(len(s)), l=1) + 1
        elif self.trunc > 0: 
            r = [i for i,ss in enumerate(sv_mass) if ss/sv_mass[-1]> self.trunc][0]+1
       
        ur, sr, vhr =  u[:, 0:r], s[0:r], vh[0:r]
        Phir = vhr
        P = np.diag(1/sr) @ ur.T 
        Phipr = P @ PhiP
        return Phir, Phipr

    def getpsi_l2(self, r): 
        """ return lambda test function 
        r: real space radius
        """
        g = lambda t: (1 - (t/r)**2)**self.p/np.sqrt(2*r*self.sum2p)
        gp = lambda t: -2*t*self.p/r**2*(1 - (t/r)**2)**(self.p-1)/np.sqrt(2*r*self.sum2p)
        return g, gp
    
    


    def getVVp_L2(self, radius):
        """Returns test funcion matrices Psi Psidot
        r: real space radius
        """
        r = radius*self.dt
        g, gp = self.getpsi_l2(r)
        t_r = np.linspace(-r, r, 2*radius+1)
        phi = g(t_r)
        phip = gp(t_r)

        centers = np.arange(radius, self.Mp1 - radius, self.gap, dtype=int)
        V = np.zeros((len(centers), self.Mp1))
        Vp = np.zeros((len(centers), self.Mp1))
        for j in range(len(centers)):
            V[j, centers[j]-radius:centers[j]+radius+1] = phi*self.dt
            Vp[j, centers[j]-radius:centers[j]+radius+1] = phip*self.dt
        return V, Vp, centers
    

    
    def get_r_c_hat(self):
        #compute derivatives at endpoints
        endpoints = self.compute_endderivative()
        #print('Done finding endpoint derivative', endpoints)

        
        #define Ftilde
        freq = np.fft.fftfreq(self.Mtilde, d=1/self.Mtilde)

        '''''
        Ftilde = np.zeros((self.M, self.Mtilde), dtype=complex) 
        for n in freq: 
            n = int(n)
            for m in range(self.M):
                tm = self.dt*m
                Ftilde[m, n] = np.exp(-2*np.pi*1j*tm*n/self.T)
        '''''
        #compute I for each dimension
        Is = []
        for i in range(self.d):
            I = self.getI(self.S, freq, self.dt, self.T, endpoints[i])
            Is.append(I)
        #print('Done compute I for each dimension')

        radii = np.arange(2, self.M//2, 1) 

        #start = time.time()
        normehat = []
        for radius in radii: 
            r = radius*self.dt
            centers = np.arange(radius, self.Mp1 - radius, self.gap, dtype=int)
            numK = len(centers)

            psihat = self.get_vecpsihat_l2(freq, r)
            eint_hat = []
            for i in range(self.d):
                eint_hat_i = (np.fft.fft(psihat*Is[i])[centers])/np.sqrt(self.T)
                eint_hat.append(eint_hat_i)
            eint_hat = np.vstack(eint_hat)
            normehat.append(np.linalg.norm(eint_hat)/np.sqrt(numK))
        normehat = np.array(normehat)
        cornerpoint = self.getcorner(np.log(normehat), radii, l=2)
        #cornerpoint = kneed.KneeLocator(radii, np.log(normehat),  curve='convex', direction="decreasing", S=1).knee
        #print('runtime fast', time.time()- start)
        return cornerpoint
    

    def getpsihatn_l2(self, r, n): 
        """Compute psihat_n 
        r: radius
        n: 
        """
        if n == 0: 
            sump = 0
            for k in range(self.p+1):
                sump = sump + math.comb(self.p, k)*(-1)**k/(2*k + 1)
            return np.sqrt(2*r/(self.sum2p*self.T))*sump
        else: 
            return 1/np.sqrt(2*self.sum2p)/r**self.p*(self.T/(n*np.pi))**self.p/np.sqrt(n)*scipy.special.gamma(self.p+1)*scipy.special.jv(self.p+1/2, 2*np.pi*n*r/self.T)


    def get_vecpsihat_l2(self, freq, r): 
        """Compute vector psihat on a set of freq 
        freq:freq
        r: radius
        """
        psihat = np.zeros(len(freq))
        for n in range(len(freq)//2+1):
            n = int(n)
            psihatn = self.getpsihatn_l2(r, n)
            psihat[n] = psihatn
            psihat[-n] = psihatn
        return psihat
    
    def es(self, s, n, dt, T, endpoints):
        suml = 0
        for l in range(2*s+1):
            #print(s, l, math.comb(2*s-1, l))
            suml = suml + math.comb(2*s, l)*(2*np.pi*n*1j/T)**(2*s-l)*endpoints[l]
        suml = suml*bernoulli(2*s)[-1]/math.factorial(2*s)*dt**(2*s)
        return suml

    def getIn(self, S,n ,dt, T, endpoints):
        sumS = 0
        for s in range(1, S+1):
            sumS = sumS + self.es(s, n, dt, T, endpoints)
        return endpoints[0] + sumS #- dt/2*(2*np.pi*n*1j/T*endpoints[0] + endpoints[1]) 

    def getI(self, S, freq ,dt, T, endpoints): 
        I = np.zeros(len(freq), dtype=complex)
        for n in freq: 
            n = int(n)
            I[n] = self.getIn(S, n, dt, T, endpoints)   
        return I

    def getFD_endpoints(self, l, mu, f, dt):
        """Compute endpts FD approx of l-th derivative with order mu accurate
        l: derivative order, l = 0, 1, ...
        mu: order of accuracy mu = 1, 2, 3... 
        f data: 1D array
        """
        if l == 0: 
            D0 = f[0]
            DT = f[-1]
        else: 
            #get forward different weights
            A = np.zeros((mu + l, mu + l))
            b = np.zeros(mu + l)
            for i in range(mu + l):
                for m in range(mu + l):
                    A[i, m] = m**i
                    if i == l:
                        b[i] = math.factorial(l)
            #weightsFD: forward difference, weightsBD: backward difference 
            weightsFD = np.linalg.solve(A, b)
            if l % 2 == 0: 
                weightsBD = weightsFD.copy()
            else: 
                weightsBD = -weightsFD.copy()[::-1]
            D0 = weightsFD.dot(f[0: mu + l])/dt**l
            DT = weightsBD.dot(f[-(mu+l):])/dt**l
        return DT - D0

    
    def compute_endderivative(self):
        endpoints = []
        for i in range(self.d):
            epts_i =[]
            if self.S == 0: 
                for l in range(2): 
                    epts_i.append(self.getFD_endpoints(l, self.mu[l], self.xobs[:, i], self.dt))
            else:
                for l in range(2*self.S+1): 
                    epts_i.append(self.getFD_endpoints(l, self.mu[l], self.xobs[:, i], self.dt))
            endpoints.append(epts_i)
        return endpoints
    

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
        
    def simulate(self, x0, t):
        tol_ode = 1e-15
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
        def blowup_event(thresh):
            def event(t, y):
                return np.linalg.norm(y, ord=np.inf) - thresh
            event.terminal = True
            event.direction = 0
            return event

        rhs_p = lambda t, x: rhs_fun(self.features, w_hat_tolist, x)
        sol = solve_ivp(rhs_p, t_span = np.array([t[0], t[-1]]), y0=x0, t_eval=t,  method='BDF', rtol=tol_ode, atol=tol_ode, events=blowup_event(1e3))
        if sol.y.shape[1] < len(t):
            print('oops')
            final_val = sol.y[:, -1][:, None]
            pad_count = len(t) - sol.y.shape[1]
            pad_vals = np.tile(final_val, (1, pad_count))
            y_full = np.hstack([sol.y, pad_vals])
        else:
            y_full = sol.y
        return y_full.T


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

    def estimate_sigma(self, f):
        k = 6
        C = self.fdcoeffF(k, 0, np.arange(-k-2, k+3))
        filter = C[:, -1]
        filter = filter / np.linalg.norm(filter, ord=2)
        filter = filter.reshape(1, -1).T
        f = f.reshape(-1, 1)
        sig = np.sqrt(np.mean(np.square(convolve2d(f, filter, mode='valid'))))
        return sig
    
    def get_RT(self, L0, L1, w, diag_reg): 
        dims = L1.shape
        if not np.all(np.all(w == 0)):
            L0 = L0 + np.reshape(np.transpose(L1, (2, 0, 1)).reshape(dims[2], -1).T @ w, (dims[0], -1))
        Cov = L0 @ L0.T
        newCov = (1-diag_reg)*Cov + diag_reg*np.eye(Cov.shape[0])
        RT = np.linalg.cholesky(newCov)
        return RT, L0, Cov, diag_reg


    def getcorner(self, Ufft, xx, l=1):
        NN = len(Ufft)
        Ufft = Ufft / max(np.abs(Ufft)) * NN
        errs = np.zeros(NN)
        
        if l == 1:
            for k in range(1, NN+1):
                L1, L2, m1, m2, b1, b2, Ufft_av1, Ufft_av2 = self.build_lines(Ufft, xx, k)
                errs[k-1] = (np.sum(np.abs((L1-Ufft_av1) / Ufft_av1)) + np.sum(np.abs((L2-Ufft_av2) / Ufft_av2))) # relative l1
        elif l == 2:
            for k in range(1, NN+1):
                L1, L2, m1, m2, b1, b2, Ufft_av1, Ufft_av2 = self.build_lines(Ufft, xx, k)
                errs[k-1] = np.sqrt(np.sum(((L1-Ufft_av1) / Ufft_av1)**2) + np.sum(((L2-Ufft_av2) / Ufft_av2)**2)) # relative l2
        elif l == 3:
            for k in range(1, NN+1):
                L1, L2, m1, m2, b1, b2, Ufft_av1, Ufft_av2 = self.build_lines(Ufft, xx, k)
                errs[k-1] = np.sum(((L1-Ufft_av1) / Ufft_av1)**2) + np.sum(((L2-Ufft_av2) / Ufft_av2)**2) # relative l2
        tstarind = np.nanargmin(errs)
        '''''
        plt.plot(xx, Ufft)
        plt.scatter(xx[tstarind], Ufft[tstarind])
        plt.show()

        plt.plot(xx, errs)
        plt.scatter(xx[tstarind], errs[tstarind])
        plt.show()
        '''''
        return xx[tstarind] 

    def lin_regress(self, U, x):
        m = (U[-1] - U[0]) / (x[-1] - x[0])
        b = U[0] - m * x[0]
        L = U[0] + m * (x - x[0])
        return m, b, L
    
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
 
    
    
    def fit_OLS(self, xobs, tobs, radius, type_rad = 0, type_tf = 0, toggle_SVD = False, gap = 1, p = 10, S = 1, mu = [1, 2, 1], Mtilde = None, diag_reg = 1e-10, trunc = 0):
        """
        Inputs:
        xobs: data
        tobs: time
        radius: grid point radius in a list (if None - compute this automatically)
        type_rad: 0 - Single-scale Local, 1 - Multi-scale Global
        type_tf: Type of test function: 0 -> L2, 1 ->L_inf
        toggle_SVD: False -> no SVD,  True -> SVD
        gap: gap between test functions
        p: order of poly tf
        S: truncation order of In
        mu: finite difference orders of accuracy

        trunc: truncation method for svd 0 -> corner point, 0 < trunc < 1 trunc% weight of singularvals
        """
        self.xobs = xobs
        self.tobs = tobs
        self.radius = radius
        self.type_rad = type_rad
        self.type_tf = type_tf
        self.toggle_SVD = toggle_SVD
        self.gap = gap
        self.p = p

        self.d = self.xobs.shape[1]
        self.dt = self.tobs[1]
        self.T = self.tobs[-1]
        self.M = len(self.tobs) - 1
        self.Mp1 = len(self.tobs)

        self.S = S
        self.mu = mu 
        if Mtilde == None: 
            Mtilde = self.M
        self.Mtilde = Mtilde
        self.trunc = trunc

        #set params
        self.iter_diff_tol = 1e-6
        self.max_iter = 100
        self.diag_reg = diag_reg
        self.pvalmin = 1e-4
        self.check_pval_it = 10
        self.tau = 1e-5


        sum2p = 0
        for k in range(2*self.p + 1):
            sum2p = sum2p + math.comb(2*self.p, k)*(-1)**k/(2*k + 1)
        self.sum2p = sum2p

        #build Theta
        xobs_cell = [self.xobs[:,i] for i in range(self.d)]
        Theta_cell = [np.vstack([y(*xobs_cell) for y in x]).T for x in self.features]

        #compute radii
        if self.radius == None: 
            if self.type_rad == 0:
                radius = [self.get_r_c_hat()]
            elif self.type_rad == 1:
                rad_min = self.get_r_min()
                radius = np.minimum(rad_min * 2**np.arange(4), self.M//2 - 1)
        self.radius = radius

        #build test function matrices
        if len(self.radius) == 1:
            Phi, PhiP, centers = self.getVVp_L2(radius[0])   
        else: 
            Phi = []
            PhiP = []
            centers = []
            for rad in radius: 
                Phi_r, PhiP_r, centers_r = self.getVVp_L2(rad)  
                Phi.append(Phi_r)
                PhiP.append(PhiP_r)
                centers.append(centers_r)
            Phi = np.vstack(Phi)
            PhiP = np.vstack(PhiP)

        #print("Phi shape", Phi.shape)
        if self.toggle_SVD == True: 
            Phi, PhiP = self.svdPhi(Phi, PhiP)
            #print("Phi shape after SVD", Phi.shape)
        self.Phi = Phi
        self.PhiP = PhiP
        self.centers = centers

        
        V_cell = [self.Phi]*self.d
        Vp_cell = [self.PhiP]*self.d
        

        # build linear system
        G0 = [Phi @ x for x, Phi in zip(Theta_cell, V_cell)]
        G0 = scipy.linalg.block_diag(*G0)
        b0 = np.hstack([-PhiP @ x for x, PhiP in zip(xobs_cell, Vp_cell)]).reshape(-1,1)
        self.G0 = G0
        self.b0 = b0
        w0 =  np.linalg.lstsq(G0, b0, rcond=None)[0].reshape(-1, 1)
        self.w_hat = w0
        return self.w_hat
    


    
