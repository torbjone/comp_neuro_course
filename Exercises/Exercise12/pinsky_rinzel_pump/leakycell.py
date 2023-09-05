import numpy as np
from math import fsum
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("error")
from .somatic_injection_current import *

class LeakyCell(): 
    """A two plus two compartment neuron model with Na+, K+, and Cl- leak currents.

    Methods
    -------
    constructor(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
        Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha)
    j_Na_s(phi_sm, E_Na_s): compute the Na+ flux across the somatic membrane
    j_K_s(phi_sm, E_K_s): compute the K+ flux across the somatic membrane
    j_Cl_s(phi_sm, E_Cl_s): compute the Cl- flux across the somatic membrane
    j_Na_d(phi_dm, E_Na_d): compute the Na+ flux across the dendritic membrane
    j_K_d(phi_dm, E_K_d): compute the K+ flux across the dendritic membrane
    j_Cl_d(phi_dm, E_Cl_d): compute the Cl- flux across the dendritic membrane
    j_k_diff(D_k, tortuosity, k_s, k_d): compute the axial diffusion flux of ion k
    j_k_drift(D_k, Z_k, tortuosity, k_s, k_d, phi_s, phi_d): compute the axial drift flux of ion k
    conductivity_k(D_k, Z_k, tortuosity, k_s, k_d): compute axial conductivity of ion k
    total_charge(k, k_res, V): calculate the total charge within volume V
    nernst_potential(Z, k_i, k_e): calculate the reversal potential of ion k
    reversal_potentials(): calculate the reversal potentials of all ion species
    membrane_potentials(): calculate the membrane potentials
    dkdt(): calculate dk/dt for all ion species k
    """

    def __init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha):
        
        # temperature [K]
        self.T = T

        # ion concentraions [mol * m**-3]
        self.Na_si = Na_si
        self.Na_se = Na_se
        self.Na_di = Na_di
        self.Na_de = Na_de
        self.K_si = K_si
        self.K_se = K_se
        self.K_di = K_di
        self.K_de = K_de
        self.Cl_si = Cl_si
        self.Cl_se = Cl_se 
        self.Cl_di = Cl_di 
        self.Cl_de = Cl_de
        self.Ca_si = Ca_si
        self.Ca_se = Ca_se 
        self.Ca_di = Ca_di 
        self.Ca_de = Ca_de
        self.k_res_si = k_res_si
        self.k_res_se = k_res_se
        self.k_res_di = k_res_di
        self.k_res_de = k_res_de
        self.free_Ca_si = 0.01*Ca_si
        self.free_Ca_di = 0.01*Ca_di

        # membrane capacitance [F * m**-2]
        self.C_sm = 3e-2 # Pinsky and Rinzel, 1994
        self.C_dm = 3e-2 # Pinsky and Rinzel, 1994
#        self.C_sm = 1e-2 # Wei et al. 2014
#        self.C_dm = 1e-2 # Wei et al. 2014
       
        # volumes and areas
        self.alpha = alpha
        self.A_s = 616e-12             # [m**2]
        self.A_d = 616e-12             # [m**2]
        self.A_i = self.alpha*self.A_s # [m**2]
        self.A_e = self.A_i/2.         # [m**2]
        self.V_si = 1437e-18           # [m**3]
        self.V_di = 1437e-18           # [m**3]
        self.V_se = 718.5e-18           # [m**3]
        self.V_de = 718.5e-18           # [m**3]
        self.dx = 667e-6               # [m]

        # diffusion constants [m**2 s**-1]
        self.D_Na = 1.33e-9 # Halnes et al. 2013
        self.D_K = 1.96e-9  # Halnes et al. 2013 
        self.D_Cl = 2.03e-9 # Halnes et al. 2013
        self.D_Ca = 0.71e-9 # Halnes et al. 2016

        # tortuosities
        self.lamda_i = 3.2 # Halnes et al. 2013
        self.lamda_e = 1.6 # Halnes et al. 2013

        # valencies
        self.Z_Na = 1.
        self.Z_K = 1.
        self.Z_Cl = -1.
        self.Z_Ca = 2.

        # constants
        self.F = 9.648e4    # [C * mol**-1]
        self.R = 8.314      # [J * mol**-1 * K**-1] 

        # conductances [S * m**-2]
        self.g_Na_leak = 0.247 # Wei et al. 2014
        self.g_K_leak = 0.5    # Wei et al. 2014
        self.g_Cl_leak = 1.0   # Wei et al. 2014

    def j_Na_s(self, phi_sm, E_Na_s):
        j = self.g_Na_leak*(phi_sm - E_Na_s) / (self.F*self.Z_Na)
        return j 

    def j_K_s(self, phi_sm, E_K_s):
        j = self.g_K_leak*(phi_sm - E_K_s) / (self.F*self.Z_K)
        return j

    def j_Cl_s(self, phi_sm, E_Cl_s):
        j = self.g_Cl_leak*(phi_sm - E_Cl_s) / (self.F*self.Z_Cl)
        return j

    def j_Na_d(self, phi_dm, E_Na_d):
        j = self.g_Na_leak*(phi_dm - E_Na_d) / (self.F*self.Z_Na) 
        return j

    def j_K_d(self, phi_dm, E_K_d):
        j = self.g_K_leak*(phi_dm - E_K_d) / (self.F*self.Z_K) 
        return j

    def j_Cl_d(self, phi_dm, E_Cl_d):
        j = self.g_Cl_leak*(phi_dm - E_Cl_d) / (self.F*self.Z_Cl) 
        return j

    def j_k_diff(self, D_k, tortuosity, k_s, k_d):
        j = - D_k * (k_d - k_s) / (tortuosity**2 * self.dx)
        return j

    def j_k_drift(self, D_k, Z_k, tortuosity, k_s, k_d, phi_s, phi_d):
        j = - D_k * self.F * Z_k * (k_d + k_s) * (phi_d - phi_s) / (2 * tortuosity**2 * self.R * self.T * self.dx)
        return j

    def conductivity_k(self, D_k, Z_k, tortuosity, k_s, k_d): 
        sigma = self.F**2 * D_k * Z_k**2 * (k_d + k_s) / (2 * self.R * self.T * tortuosity**2)
        return sigma

    def total_charge(self, k, k_res, V):
        Z_k = [self.Z_Na, self.Z_K, self.Z_Cl, self.Z_Ca]
        q = 0.0
        for i in range(0, 4):
            q += Z_k[i]*k[i]
        q = self.F*(q + k_res)*V
        return q

    def nernst_potential(self, Z, k_i, k_e):
        E = self.R*self.T / (Z*self.F) * np.log(k_e / k_i)
        #E = 26.64e-3 * np.log(k_e / k_i) / Z
        return E

    def reversal_potentials(self):
        E_Na_s = self.nernst_potential(self.Z_Na, self.Na_si, self.Na_se)
        E_Na_d = self.nernst_potential(self.Z_Na, self.Na_di, self.Na_de)
        E_K_s = self.nernst_potential(self.Z_K, self.K_si, self.K_se)
        E_K_d = self.nernst_potential(self.Z_K, self.K_di, self.K_de)
        E_Cl_s = self.nernst_potential(self.Z_Cl, self.Cl_si, self.Cl_se)
        E_Cl_d = self.nernst_potential(self.Z_Cl, self.Cl_di, self.Cl_de)
        E_Ca_s = self.nernst_potential(self.Z_Ca, self.free_Ca_si, self.Ca_se)
        E_Ca_d = self.nernst_potential(self.Z_Ca, self.free_Ca_di, self.Ca_de)
        return E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d

    def membrane_potentials(self):
        I_i_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_i, self.K_si, self.K_di) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_i, self.Cl_si, self.Cl_di) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di))
        I_e_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_e, self.K_se, self.K_de) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_e, self.Cl_se, self.Cl_de) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_e, self.Ca_se, self.Ca_de))

        sigma_i = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_i, self.K_si, self.K_di) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, self.Cl_si, self.Cl_di) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di)
        sigma_e = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_e, self.K_se, self.K_de) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_e, self.Cl_se, self.Cl_de) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_e, self.Ca_se, self.Ca_de)

        q_di = self.total_charge([self.Na_di, self.K_di, self.Cl_di, self.Ca_di], self.k_res_di, self.V_di)
        q_si = self.total_charge([self.Na_si, self.K_si, self.Cl_si, self.Ca_si], self.k_res_si, self.V_si)

        phi_di = q_di / (self.C_dm * self.A_d)
        phi_se = (phi_di - self.dx * I_i_diff / sigma_i - self.A_e * self.dx * I_e_diff / (self.A_i * sigma_i) - q_si / (self.C_sm * self.A_s)) / (1 + self.A_e*sigma_e/(self.A_i*sigma_i))
        phi_si = q_si / (self.C_sm * self.A_s) + phi_se
        phi_de = 0.
        phi_sm = phi_si - phi_se
        phi_dm = phi_di - phi_de

        return phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm

    def dkdt(self):
       
        phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm  = self.membrane_potentials()
        E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = self.reversal_potentials()

        j_Na_sm = self.j_Na_s(phi_sm, E_Na_s)
        j_K_sm = self.j_K_s(phi_sm, E_K_s)
        j_Cl_sm = self.j_Cl_s(phi_sm, E_Cl_s)

        j_Na_dm = self.j_Na_d(phi_dm, E_Na_d)
        j_K_dm = self.j_K_d(phi_dm, E_K_d)    
        j_Cl_dm = self.j_Cl_d(phi_dm, E_Cl_d)

        j_Na_i = self.j_k_diff(self.D_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_i, self.Na_si, self.Na_di, phi_si, phi_di) 
        j_K_i = self.j_k_diff(self.D_K, self.lamda_i, self.K_si, self.K_di) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_i, self.K_si, self.K_di, phi_si, phi_di)
        j_Cl_i = self.j_k_diff(self.D_Cl, self.lamda_i, self.Cl_si, self.Cl_di) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_i, self.Cl_si, self.Cl_di, phi_si, phi_di)
        j_Ca_i = self.j_k_diff(self.D_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di, phi_si, phi_di)

        j_Na_e = self.j_k_diff(self.D_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_e, self.Na_se, self.Na_de, phi_se, phi_de)
        j_K_e = self.j_k_diff(self.D_K, self.lamda_e, self.K_se, self.K_de) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_e, self.K_se, self.K_de, phi_se, phi_de)
        j_Cl_e = self.j_k_diff(self.D_Cl, self.lamda_e, self.Cl_se, self.Cl_de) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_e, self.Cl_se, self.Cl_de, phi_se, phi_de)
        j_Ca_e = self.j_k_diff(self.D_Ca, self.lamda_e, self.Ca_se, self.Ca_de) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_e, self.Ca_se, self.Ca_de, phi_se, phi_de)

        dNadt_si = -j_Na_sm*(self.A_s / self.V_si) - j_Na_i*(self.A_i / self.V_si)
        dNadt_di = -j_Na_dm*(self.A_d / self.V_di) + j_Na_i*(self.A_i / self.V_di)
        dNadt_se = j_Na_sm*(self.A_s / self.V_se) - j_Na_e*(self.A_e / self.V_se)
        dNadt_de = j_Na_dm*(self.A_d / self.V_de) + j_Na_e*(self.A_e / self.V_de)

        dKdt_si = -j_K_sm*(self.A_s / self.V_si) - j_K_i*(self.A_i / self.V_si)
        dKdt_di = -j_K_dm*(self.A_d / self.V_di) + j_K_i*(self.A_i / self.V_di)
        dKdt_se = j_K_sm*(self.A_s / self.V_se) - j_K_e*(self.A_e / self.V_se)
        dKdt_de = j_K_dm*(self.A_d / self.V_de) + j_K_e*(self.A_e / self.V_de)

        dCldt_si = -j_Cl_sm*(self.A_s / self.V_si) - j_Cl_i*(self.A_i / self.V_si)
        dCldt_di = -j_Cl_dm*(self.A_d / self.V_di) + j_Cl_i*(self.A_i / self.V_di)
        dCldt_se = j_Cl_sm*(self.A_s / self.V_se) - j_Cl_e*(self.A_e / self.V_se)
        dCldt_de = j_Cl_dm*(self.A_d / self.V_de) + j_Cl_e*(self.A_e / self.V_de)

        dCadt_si = - j_Ca_i*(self.A_i / self.V_si)
        dCadt_di = j_Ca_i*(self.A_i / self.V_di)
        dCadt_se = - j_Ca_e*(self.A_e / self.V_se)
        dCadt_de = j_Ca_e*(self.A_e / self.V_de)

        dresdt_si = 0
        dresdt_di = 0
        dresdt_se = 0
        dresdt_de = 0

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de, dresdt_si, dresdt_se, dresdt_di, dresdt_de

if __name__ == "__main__":

    T = 309.14
    alpha = 1.

    Na_si0 = 15.
    Na_se0 = 145.
    K_si0 = 100.
    K_se0 = 3.
    Cl_si0 = 5.
    Cl_se0 = 134.
    Ca_si0 = 0.001
    Ca_se0 = 1.1

    Na_di0 = 20.
    Na_de0 = 150.
    K_di0 = 100.
    K_de0 = 3.
    Cl_di0 = 5.
    Cl_de0 = 134.
    Ca_di0 = 0.001
    Ca_de0 = 1.1

    k_res_si = Cl_si0 - Na_si0 - K_si0 - 2*Ca_si0
    k_res_se = Cl_se0 - Na_se0 - K_se0 - 2*Ca_se0
    k_res_di = Cl_di0 - Na_di0 - K_di0 - 2*Ca_di0
    k_res_de = Cl_de0 - Na_de0 - K_de0 - 2*Ca_de0

    def dkdt(t,k):

        Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de = k

        my_cell = LeakyCell(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha)

        dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCadt_si, dCadt_se, dCadt_di, dCadt_de, dresdt_si, dresdt_se, dresdt_di, dresdt_de = my_cell.dkdt()

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, \
            dCadt_si, dCadt_se, dCadt_di, dCadt_de
    
    start_time = time.time()
    t_span = (0, 1)

    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0]

    init_cell = LeakyCell(T, Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, k_res_si, k_res_se, k_res_di, k_res_de, alpha)

    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = init_cell.membrane_potentials()
    
    E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = init_cell.reversal_potentials()

    q_si = init_cell.total_charge([init_cell.Na_si, init_cell.K_si, init_cell.Cl_si, init_cell.Ca_si], init_cell.k_res_si, init_cell.V_si)
    q_se = init_cell.total_charge([init_cell.Na_se, init_cell.K_se, init_cell.Cl_se, init_cell.Ca_se], init_cell.k_res_se, init_cell.V_se)        
    q_di = init_cell.total_charge([init_cell.Na_di, init_cell.K_di, init_cell.Cl_di, init_cell.Ca_di], init_cell.k_res_di, init_cell.V_di)
    q_de = init_cell.total_charge([init_cell.Na_de, init_cell.K_de, init_cell.Cl_de, init_cell.Ca_de], init_cell.k_res_de, init_cell.V_de)
    print("----------------------------")
    print("Initial values")
    print("----------------------------")
    print("initial total charge(C): ", q_si + q_se + q_di + q_de)
    print("Q_si (C):", q_si)
    print("Q_se (C): ", q_se)
    print("Q_di (C):", q_di)
    print("Q_de (C): ", q_de)
    print("----------------------------")
    print('phi_si: ', phi_si)
    print('phi_se: ', phi_se)
    print('phi_di: ', phi_di)
    print('phi_de: ', phi_de)
    print('phi_sm: ', phi_sm)
    print('phi_dm: ', phi_dm)
    print('E_Na_s: ', E_Na_s)
    print('E_Na_d: ', E_Na_d)
    print('E_K_s: ', E_K_s)
    print('E_K_d: ', E_K_d)
    print('E_Cl_s: ', E_Cl_s)
    print('E_Cl_d: ', E_Cl_d)
    print('E_Ca_s: ', E_Ca_s)
    print('E_Ca_d: ', E_Ca_d)
    print("----------------------------")

    sol = solve_ivp(dkdt, t_span, k0, max_step=1e-4)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de = sol.y
    t = sol.t

    my_cell = LeakyCell(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha)
    
    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = my_cell.membrane_potentials()
    
    E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = my_cell.reversal_potentials()

    q_si = my_cell.total_charge([my_cell.Na_si[-1], my_cell.K_si[-1], my_cell.Cl_si[-1], my_cell.Ca_si[-1]], my_cell.k_res_si, my_cell.V_si)
    q_se = my_cell.total_charge([my_cell.Na_se[-1], my_cell.K_se[-1], my_cell.Cl_se[-1], my_cell.Ca_se[-1]], my_cell.k_res_se, my_cell.V_se)        
    q_di = my_cell.total_charge([my_cell.Na_di[-1], my_cell.K_di[-1], my_cell.Cl_di[-1], my_cell.Ca_di[-1]], my_cell.k_res_di, my_cell.V_di)
    q_de = my_cell.total_charge([my_cell.Na_de[-1], my_cell.K_de[-1], my_cell.Cl_de[-1], my_cell.Ca_de[-1]], my_cell.k_res_de, my_cell.V_de)
    print("Final values")
    print("----------------------------")
    print("total charge at the end (C): ", q_si + q_se + q_di + q_de)
    print("Q_si (C):", q_si)
    print("Q_se (C): ", q_se)
    print("Q_di (C):", q_di)
    print("Q_de (C): ", q_de)

    print("----------------------------")
    print('elapsed time: ', round(time.time() - start_time, 1), 'seconds')

    plt.plot(t, phi_sm, '-', label='Vs')
    plt.plot(t, phi_dm, '-', label='Vd')
    plt.title('Membrane potentials')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    plt.plot(t, E_Na_s, label='E_Na')
    plt.plot(t, E_K_s, label='E_K')
    plt.plot(t, E_Cl_s, label='E_Cl')
    plt.title('Reversal potentials soma')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    plt.plot(t, E_Na_d, label='E_Na')
    plt.plot(t, E_K_d, label='E_K')
    plt.plot(t, E_Cl_d, label='E_Cl')
    plt.title('Reversal potentials dendrite')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    plt.plot(t, Na_si, label='Na_si')
    plt.plot(t, Na_se, label='Na_se')
    plt.plot(t, Na_di, label='Na_di')
    plt.plot(t, Na_de, label='Na_de')
    plt.title('Sodium concentrations')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    plt.plot(t, K_si, label='K_si')
    plt.plot(t, K_se, label='K_se')
    plt.plot(t, K_di, label='K_di')
    plt.plot(t, K_de, label='K_de')
    plt.title('Potassium concentrations')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    plt.plot(t, Cl_si, label='Cl_si')
    plt.plot(t, Cl_se, label='Cl_se')
    plt.plot(t, Cl_di, label='Cl_di')
    plt.plot(t, Cl_de, label='Cl_de')
    plt.title('Chloride concentrations')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    plt.plot(t, Ca_si, label='Ca_si')
    plt.plot(t, Ca_se, label='Ca_se')
    plt.plot(t, Ca_di, label='Ca_di')
    plt.plot(t, Ca_de, label='Ca_de')
    plt.title('Calsium concentrations')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()
