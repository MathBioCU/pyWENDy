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


class WENDy:
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
            if self.type_rad == 0: # SL Singlescale Local 
                radius = [self.get_r_c_hat()]
            elif self.type_rad == 1: # MG Multiscale Global
                rad_min = self.get_r_min()
                radius = np.minimum(rad_min * 2**np.arange(4), self.M//2 - 1)
            elif self.type_rad == 2: # SM Spectral Matching
                radius, p = self.get_rad_spectral_matching()
                radius = [ np.minimum(radius, self.M//2 - 1- self.gap) ]
                self.p = p
                sum2p = 0
                for k in range(2*self.p + 1):
                    sum2p = sum2p + math.comb(2*self.p, k)*(-1)**k/(2*k + 1)
                self.sum2p = sum2p
                #print(p)
        self.radius = radius

        #build test function matrices
        if len(self.radius) == 1:
            Phi, PhiP, centers = WENDy.getVVp_L2(radius[0], self.dt, self.Mp1, self.gap, self.p, self.sum2p)   
        else: 
            Phi = []
            PhiP = []
            centers = []
            for rad in radius: 
                Phi_r, PhiP_r, centers_r = WENDy.getVVp_L2(rad, self.dt, self.Mp1, self.gap, self.p, self.sum2p)   
                Phi.append(Phi_r)
                PhiP.append(PhiP_r)
                centers.append(centers_r)
            Phi = np.vstack(Phi)
            PhiP = np.vstack(PhiP)

        #print("Phi shape", Phi.shape)
        if self.toggle_SVD == True: 
            Phi, PhiP = WENDy.svdPhi(Phi, PhiP, self.trunc)
            #print("Phi shape after SVD", Phi.shape)
        self.Phi = Phi
        self.PhiP = PhiP
        self.centers = centers

        V_cell = [self.Phi]*self.d
        Vp_cell = [self.PhiP]*self.d

        # build initial linear system
        G0 = [Phi @ x for x, Phi in zip(Theta_cell, V_cell)]
        G0 = scipy.linalg.block_diag(*G0)
        b0 = np.hstack([-PhiP @ x for x, PhiP in zip(xobs_cell, Vp_cell)]).reshape(-1,1)
        w0, nits_w0 =  WENDy.solve_lsqr(G0, b0)
        
   
        # estimate noise variance
        sig_ests = np.array([WENDy.estimate_sigma(self.xobs[:, i]) for i in range(self.d)])
        sigma_scale = np.kron(sig_ests, np.ones(self.Mp1))

        # .... and build initial covariance
        Jac = WENDy.build_Jac_sym(self.features, self.xobs)
        param_length_vec = np.array([len(x) for x in self.features])
        L0, L1 = WENDy.get_Lfac(Jac,param_length_vec,V_cell,Vp_cell)
        L0 = L0 * sigma_scale[np.newaxis, :]
        L1 = L1 * sigma_scale[np.newaxis, :, np.newaxis]



        pvals_list = []
        nits_list = []
        w_hat_its = []

        res = G0 @ w0 - b0
        _, pvals = scipy.stats.shapiro(res)
        pvals_list.append(pvals)

        nits_list.append(nits_w0)

        w_hat = w0.reshape(-1,1)
        w_hat_its.append(w_hat)


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
            
            G = scipy.linalg.solve_triangular(RT, G0,  lower=True)
            b = scipy.linalg.solve_triangular(RT, b0,  lower=True)
            w_hat, nits_w_hat =  WENDy.solve_lsqr(G, b, x0=w_hat.reshape(-1))

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
            nits_list.append(nits_w_hat)

        if pval < self.pvalmin:
            print('error: WENDy iterates diverged')
            ind = np.argmax(pvals_list)
            w_hat = w_hat_its[ind]
            w_hat_its.append(w_hat)

        self.w_hat = w_hat
        self.w = self.w_hat
        self.pvals_list = pvals_list
        self.w_hat_its = w_hat_its
        self.nits_list = nits_list
        return self.w_hat

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
            if self.type_rad == 0: #Singlescale Local
                radius = [self.get_r_c_hat()]
            elif self.type_rad == 1: #Multiscale Global
                rad_min = self.get_r_min()
                radius = np.minimum(rad_min * 2**np.arange(4), self.M//2 - 1)
            elif self.type_rad == 2: # SM Spectral Matching
                radius, p = self.get_rad_spectral_matching()
                radius = [ np.minimum(radius, self.M//2 - 1- self.gap) ]
                self.p = p
                sum2p = 0
                for k in range(2*self.p + 1):
                    sum2p = sum2p + math.comb(2*self.p, k)*(-1)**k/(2*k + 1)
                self.sum2p = sum2p
                #print(p)
        self.radius = radius

        #build test function matrices
        if len(self.radius) == 1:
            Phi, PhiP, centers = WENDy.getVVp_L2(radius[0], self.dt, self.Mp1, self.gap, self.p, self.sum2p)   
        else: 
            Phi = []
            PhiP = []
            centers = []
            for rad in radius: 
                Phi_r, PhiP_r, centers_r = WENDy.getVVp_L2(rad, self.dt, self.Mp1, self.gap, self.p, self.sum2p)   
                Phi.append(Phi_r)
                PhiP.append(PhiP_r)
                centers.append(centers_r)
            Phi = np.vstack(Phi)
            PhiP = np.vstack(PhiP)

        #print("Phi shape", Phi.shape)
        if self.toggle_SVD == True: 
            Phi, PhiP = WENDy.svdPhi(Phi, PhiP, self.trunc)
            #print("Phi shape after SVD", Phi.shape)
        self.Phi = Phi
        self.PhiP = PhiP
        self.centers = centers

        V_cell = [self.Phi]*self.d
        Vp_cell = [self.PhiP]*self.d

        # build initial linear system
        G0 = [Phi @ x for x, Phi in zip(Theta_cell, V_cell)]
        G0 = scipy.linalg.block_diag(*G0)
        b0 = np.hstack([-PhiP @ x for x, PhiP in zip(xobs_cell, Vp_cell)]).reshape(-1,1)
        w0, nits_w0 =  WENDy.solve_lsqr(G0, b0)
        self.w0 = w0
        self.w = w0
        return self.w0



    def get_r_c_hat(self):
        #compute derivatives at endpoints
        endpoints = WENDy.compute_endderivative(self.mu, self.S, self.xobs, self.dt)
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
            I = WENDy.getI(self.S, freq, self.dt, self.T, endpoints[i])
            Is.append(I)
        #print('Done compute I for each dimension')

        radii = np.arange(2, self.M//2, 1) 

        #start = time.time()
        normehat = []
        for radius in radii: 
            r = radius*self.dt
            centers = np.arange(radius, self.Mp1 - radius, self.gap, dtype=int)
            numK = len(centers)

            psihat = WENDy.get_vecpsihat_l2(freq, r, self.p, self.T, self.sum2p)
            eint_hat = []
            for i in range(self.d):
                eint_hat_i = (np.fft.fft(psihat*Is[i])[centers])/np.sqrt(self.T)
                eint_hat.append(eint_hat_i)
            eint_hat = np.vstack(eint_hat)
            normehat.append(np.linalg.norm(eint_hat)/np.sqrt(numK))
        normehat = np.array(normehat)
        cornerpoint = WENDy.getcorner(np.log(normehat), radii, l=2)
        #cornerpoint = kneed.KneeLocator(radii, np.log(normehat),  curve='convex', direction="decreasing", S=1).knee
        #print('runtime fast', time.time()- start)
        return cornerpoint

    def get_r_min(self):
        radii = np.arange(2, self.M//2, 1) 
        s=3
        ntilde = int(np.floor(self.M/s))

        erms = []
        for radius in radii:
            Phi, _, _ = WENDy.getVVp_L2(radius, self.dt, self.Mp1, self.gap, self.p, self.sum2p)
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
        
        rad = WENDy.getcorner(np.log(erms), radii, l=2)
        return rad
    

    def get_rad_spectral_matching(self):
        Uffts = []
        for i in range(self.d):
            Ufft = abs(np.fft.fft(self.xobs[:, i]))
            Uffts.append(Ufft)
        Uffts = np.vstack(Uffts)
        Ufft = np.average(Uffts, axis=0)

        halffreq = np.arange(-(self.Mp1)//2, 1, 1)
        halfUfft = Ufft[halffreq]
        k = -WENDy.getcorner(np.cumsum(halfUfft), halffreq, l=2)

        tauhat = 2
        tau = 1e-10
        F = lambda m: np.log((2*m-1)/(m**2))*(4*np.pi**2*m**2*k**2 - 3*self.Mp1**2*tauhat**2) - 2*self.Mp1**2*tauhat**2*np.log(tau)
        const = np.sqrt(3)/np.pi*self.Mp1/2/k*tauhat
        left_bound = const
        right_bound = np.sqrt(1- 8/np.sqrt(3)*np.log(tau))*const
        m_star = scipy.optimize.brentq(F, left_bound, right_bound)
        p =( ((2*np.pi*k*m_star)/(tauhat*self.Mp1))**2 - 3)/2
        return int(m_star), int(p)
        



    def simulate(self, x0, t):
        tol_ode = 1e-13
        w_hat_tolist = []
        count = 0
        for i in range(len(self.features)): 
            a = self.features[i]
            coef = []
            for j in range(len(a)):
                coef.append(self.w[count+j][0])
            count = count + len(a)
            w_hat_tolist.append(coef)

        rhs_p = lambda t, x: WENDy.rhs_fun(self.features, w_hat_tolist, x)
        sol = solve_ivp(rhs_p, t_span = np.array([t[0], t[-1]]), y0=x0, t_eval=t,  method='BDF', rtol=tol_ode, atol=tol_ode, events=WENDy.blowup_event(1e3))
        if sol.y.shape[1] < len(t):
            print('oops')
            final_val = sol.y[:, -1][:, None]
            pad_count = len(t) - sol.y.shape[1]
            pad_vals = np.tile(final_val, (1, pad_count))
            y_full = np.hstack([sol.y, pad_vals])
        else:
            y_full = sol.y
        return y_full.T
    

    @staticmethod
    def blowup_event(thresh):
        def event(t, y):
            return np.linalg.norm(y, ord=np.inf) - thresh
        event.terminal = True
        event.direction = 0
        return event

    
    @staticmethod
    def rhs_fun(features, params, x):
        nstates = len(x)
        x = tuple(x)
        dx = np.zeros(nstates)
        for i in range(nstates):
            dx[i] = np.sum([f(*x)*p for f, p in zip(features[i], params[i])])
        return dx

    @staticmethod
    def get_RT(L0, L1, w, diag_reg): 
        dims = L1.shape
        if not np.all(np.all(w == 0)):
            L0 = L0 + np.reshape(np.transpose(L1, (2, 0, 1)).reshape(dims[2], -1).T @ w, (dims[0], -1))
        Cov = L0 @ L0.T
        newCov = (1-diag_reg)*Cov + diag_reg*np.eye(Cov.shape[0])
        RT = np.linalg.cholesky(newCov)
        return RT, L0, Cov, diag_reg


    @staticmethod
    def get_Lfac(Jac_mat, Js, V_cell, Vp_cell):
        """ 
        #This computes the matrices L0 (phi dot term) L1 (with jacobian matrix)
        return L0 shape Kd x (M+1)d 
        L1: shape Kd x (M+1)d x J

        """
        J, d, Mp1 = Jac_mat.shape
        Jac_mat = np.transpose(Jac_mat, (1, 2, 0))
        eq_inds = np.where(Js)[0]
        num_eq = len(eq_inds)
        L0 = block_diag(*Vp_cell) #shape Kd x (M+1)d
        L1 = np.zeros((L0.shape[0], d*Mp1, J))
        Ktot = 0
        Jtot = 0
        for i in range(num_eq):
            K, _ = V_cell[i].shape
            J = Js[eq_inds[i]]
            for ell in range(d):
                m = np.expand_dims(Jac_mat[ell, :, Jtot+(np.arange(J))].T, axis = 0)
                n = V_cell[i][:, :, np.newaxis]
                ixgrid = np.ix_(range(Ktot, Ktot + K), range(ell*Mp1, (ell+1)*Mp1), range(Jtot, Jtot + J))
                L1[ixgrid] = m*n
            Ktot = Ktot + K
            Jtot = Jtot + J
        return L0, L1

    @staticmethod
    def build_Jac_sym(features, xobs):
        """ 
        #This computes the Jacobian of every feature with respect to every state variable, evaluated at every time step. 
         return Jac_{j, i, m} = partial_fj/partial_ui (t_m)
        #Shape (J, d, M+1)  
        """
        Mp1, d = xobs.shape
        features = [f for f_list in features for f in f_list]
        J = len(features)
        Jac_mat = np.zeros((J, d, Mp1))

        # Create the symbolic variables
        args = symbols('x0:%d' % d)
    
        def diff_lambda(f, var):
            #args = symbols('x0:%d' % f.__code__.co_argcount)
            return sympify(diff(f(*args), var))

        for j in range(J):
            f = features[j]
            for state in range(d): 
                g = diff_lambda(f, args[state])
                G = lambdify(args, g, 'numpy')
                for i in range(Mp1):
                    x_val = xobs[i, :]
                    z = G(*x_val)
                    Jac_mat[j, state , i] =  z
        return Jac_mat


    @staticmethod
    def svdPhi(Phi, PhiP, trunc): 
        """ return orthogonal test functions
        trunc: truncation mode for svd:  0 -> corner point, 0 < trunc < 1 trunc% weight of singularvals
        """
        u, s, vh = np.linalg.svd(Phi)
        sv_mass = np.cumsum(s)
        if trunc == 0:
            r = WENDy.getcorner(sv_mass/np.sum(s), np.arange(len(s)), l=1) + 1
        elif trunc > 0: 
            r = [i for i,ss in enumerate(sv_mass) if ss/sv_mass[-1]> trunc][0]+1
       
        ur, sr, vhr =  u[:, 0:r], s[0:r], vh[0:r]
        Phir = vhr
        P = np.diag(1/sr) @ ur.T 
        Phipr = P @ PhiP
        return Phir, Phipr


    @staticmethod
    def getVVp_L2(radius, dt, Mp1, gap, p, sum2p ):
        """Returns test funcion matrices Psi Psidot
        radius: grid point radius
        Mp1: M+1 total time points
        gap: 
        """
        r = radius*dt #real space radius
        g, gp = WENDy.getpsi_l2(r, p, sum2p)
        t_r = np.linspace(-r, r, 2*radius+1)
        phi = g(t_r)
        phip = gp(t_r)

        centers = np.arange(radius, Mp1 - radius, gap, dtype=int)
        V = np.zeros((len(centers), Mp1))
        Vp = np.zeros((len(centers), Mp1))
        for j in range(len(centers)):
            V[j, centers[j]-radius:centers[j]+radius+1] = phi*dt
            Vp[j, centers[j]-radius:centers[j]+radius+1] = phip*dt
        return V, Vp, centers
    

    @staticmethod
    def getpsi_l2(r, p, sum2p): 
        """ return lambda test function
        r: real space radius, p: poly order
        """
        g = lambda t: (1 - (t/r)**2)**p/np.sqrt(2*r*sum2p)
        gp = lambda t: -2*t*p/r**2*(1 - (t/r)**2)**(p-1)/np.sqrt(2*r*sum2p)
        return g, gp
    


    @staticmethod
    def getcorner(Ufft, xx, l=1):
        NN = len(Ufft)
        Ufft = Ufft / max(np.abs(Ufft)) * NN
        errs = np.zeros(NN)
        
        if l == 1:
            for k in range(1, NN+1):
                L1, L2, m1, m2, b1, b2, Ufft_av1, Ufft_av2 = WENDy.build_lines(Ufft, xx, k)
                errs[k-1] = (np.sum(np.abs((L1-Ufft_av1) / Ufft_av1)) + np.sum(np.abs((L2-Ufft_av2) / Ufft_av2))) # relative l1
        elif l == 2:
            for k in range(1, NN+1):
                L1, L2, m1, m2, b1, b2, Ufft_av1, Ufft_av2 = WENDy.build_lines(Ufft, xx, k)
                errs[k-1] = np.sqrt(np.sum(((L1-Ufft_av1) / Ufft_av1)**2) + np.sum(((L2-Ufft_av2) / Ufft_av2)**2)) # relative l2
        elif l == 3:
            for k in range(1, NN+1):
                L1, L2, m1, m2, b1, b2, Ufft_av1, Ufft_av2 = WENDy.build_lines(Ufft, xx, k)
                errs[k-1] = np.sum(((L1-Ufft_av1) / Ufft_av1)**2) + np.sum(((L2-Ufft_av2) / Ufft_av2)**2) # relative l2
        tstarind = np.nanargmin(errs)

        #print(tstarind)
        #plt.plot(xx, errs)
        #plt.scatter(xx[tstarind], errs[tstarind])
        #plt.show()

        return xx[tstarind] 
    

    @staticmethod
    def lin_regress(U, x):
        m = (U[-1] - U[0]) / (x[-1] - x[0])
        b = U[0] - m * x[0]
        L = U[0] + m * (x - x[0])
        return m, b, L
    
    @staticmethod
    def build_lines(Ufft, xx, k):
        NN = len(Ufft)
        subinds1 = np.arange(0, k)
        subinds2 = np.arange(k-1, NN)
        Ufft_av1 = Ufft[subinds1]
        Ufft_av2 = Ufft[subinds2]
        xx = np.array(xx)
        m1, b1, L1 = WENDy.lin_regress(Ufft_av1, xx[subinds1])
        m2, b2, L2 = WENDy.lin_regress(Ufft_av2, xx[subinds2])
        return L1, L2, m1, m2, b1, b2, Ufft_av1, Ufft_av2




    @staticmethod
    def get_vecpsihat_l2(freq, r, p, T, sum2p): 
        """Compute vector psihat on a set of freq 
        freq:freq
        r: radius, p: poly order, T: end time, sum2p: ...
        """
        psihat = np.zeros(len(freq))
        for n in range(len(freq)//2+1):
            n = int(n)
            psihatn = WENDy.getpsihatn_l2(r, n, p, T, sum2p)
            psihat[n] = psihatn
            psihat[-n] = psihatn
        return psihat

    @staticmethod
    def getpsihatn_l2(r, n, p, T, sum2p): 
        """Compute psihat_n 
        r: radius
        n: frequency
        """
        if n == 0: 
            sump = 0
            for k in range(p+1):
                sump = sump + math.comb(p, k)*(-1)**k/(2*k + 1)
            return np.sqrt(2*r/(sum2p*T))*sump
        else: 
            return 1/np.sqrt(2*sum2p)/r**p*(T/(n*np.pi))**p/np.sqrt(n)*scipy.special.gamma(p+1)*scipy.special.jv(p+1/2, 2*np.pi*n*r/T)


    @staticmethod
    def getI(S, freq ,dt, T, endpoints): 
        I = np.zeros(len(freq), dtype=complex)
        for n in freq: 
            n = int(n)
            I[n] = WENDy.getIn(S, n, dt, T, endpoints)   
        return I
    
    @staticmethod
    def getIn(S,n ,dt, T, endpoints):
        sumS = 0
        for s in range(1, S+1):
            sumS = sumS + WENDy.es(s, n, dt, T, endpoints)
        return endpoints[0] + sumS #- dt/2*(2*np.pi*n*1j/T*endpoints[0] + endpoints[1]) 
    
    def es(s, n, dt, T, endpoints):
        suml = 0
        for l in range(2*s+1):
            #print(s, l, math.comb(2*s-1, l))
            suml = suml + math.comb(2*s, l)*(2*np.pi*n*1j/T)**(2*s-l)*endpoints[l]
        suml = suml*bernoulli(2*s)[-1]/math.factorial(2*s)*dt**(2*s)
        return suml


    @staticmethod
    def compute_endderivative(mu, S, xobs, dt):
        d = xobs.shape[1]
        endpoints = []
        for i in range(d):
            epts_i =[]
            if S == 0: 
                for l in range(2): 
                    epts_i.append(WENDy.getFD_endpoints(l, mu[l], xobs[:, i], dt))
            else:
                for l in range(2*S+1): 
                    epts_i.append(WENDy.getFD_endpoints(l, mu[l], xobs[:, i], dt))
            endpoints.append(epts_i)
        return endpoints
    

    @staticmethod
    def getFD_endpoints(l, mu, f, dt):
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
    

    @staticmethod
    def estimate_sigma(f):
        k = 6
        C = WENDy.fdcoeffF(k, 0, np.arange(-k-2, k+3))
        filter = C[:, -1]
        filter = filter / np.linalg.norm(filter, ord=2)
        filter = filter.reshape(1, -1).T
        f = f.reshape(-1, 1)
        sig = np.sqrt(np.mean(np.square(convolve2d(f, filter, mode='valid'))))
        return sig
    
    @staticmethod
    def fdcoeffF(k, xbar, x):
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
    

      
    @staticmethod
    def solve_lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=None, x0=None):
        """
        Solve min ||Ax - b||_2 using LSQR.

        Parameters
        ----------
        A : ndarray, sparse matrix, or LinearOperator, shape (m, n)
        b : ndarray, shape (m,)
        atol, btol : float
            Stopping tolerances for LSQR.
        iter_lim : int or None
            Maximum number of iterations.
        x0 : ndarray or None
            Optional initial guess.

        Returns
        -------
        x : ndarray, shape (n,)
            Approximate least-squares solution.
        info : dict
            Extra solver information.
        """
        result = scipy.sparse.linalg.lsqr(A, b, atol=atol, btol=btol, iter_lim=iter_lim, x0=x0)

        x = result[0]
        nits = result[2]
        '''''
        info = {
            "istop": result[1],       # reason for stopping
            "iterations": result[2],  # number of iterations
            "r1norm": result[3],      # ||b - A x||
            "r2norm": result[4],      # regularized residual norm
            "anorm": result[5],       # estimate of ||A||
            "acond": result[6],       # estimate of cond(A)
            "arnorm": result[7],      # ||A^T r||
            "xnorm": result[8],       # ||x||
        }
        '''''
        return x.reshape(-1, 1), nits
        







