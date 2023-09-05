from .leakycell import LeakyCell
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from .somatic_injection_current import *

class Pump(LeakyCell):
    """A two plus two compartment neuron model with Na, K and Cl leak currents and pumps.

    Attributes
    ----------
    LeakyCell (Class)

    Methods
    -------
    constructor(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
        Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha)
    j_pump(Na_i, K_e): compute the Na+/K+ pump flux across given membrane
    j_kcc2(K_i, K_e, Cl_i, Cl_e): compute the K+/Cl- co-transporter flux across given membrane
    j_nkcc1(self, Na_i, Na_e, K_i, K_e, Cl_i, Cl_e): compute the Na+/K+/Cl- co-transporter flux across given membrane
    j_Na_s(phi_sm, E_Na_s): compute the Na+ flux across the somatic membrane
    j_K_s(phi_sm, E_K_s): compute the K+ flux across the somatic membrane
    j_Cl_s(phi_sm, E_Cl_s): compute the Cl- flux across the somatic membrane
    j_Na_d(phi_dm, E_Na_d): compute the Na+ flux across the dendritic membrane
    j_K_d(phi_dm, E_K_d): compute the K+ flux across the dendritic membrane
    j_Cl_d(phi_dm, E_Cl_d): compute the Cl- flux across the dendritic membrane
    """

    def __init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha):
        LeakyCell.__init__(self, T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha)
        self.rho = 1.87e-6
        self.U_kcc2 = 7.00e-7
        self.U_nkcc1 = 2.33e-7

    def j_pump(self, Na_i, K_e):
        j = (self.rho / (1.0 + np.exp((25. - Na_i)/3.))) * (1.0 / (1.0 + np.exp(3.5 - K_e)))
        return j

    def j_kcc2(self, K_i, K_e, Cl_i, Cl_e):
        j = self.U_kcc2 * np.log(K_i*Cl_i/(K_e*Cl_e))
        return j
    
    def j_nkcc1(self, Na_i, Na_e, K_i, K_e, Cl_i, Cl_e):
        j = self.U_nkcc1 * (1 / (1 + np.exp(16 - K_e))) * (np.log(K_i*Cl_i/(K_e*Cl_e)) + np.log(Na_i*Cl_i/(Na_e*Cl_e)))
        return j

    def j_Na_s(self, phi_sm, E_Na_s):
        j = LeakyCell.j_Na_s(self, phi_sm, E_Na_s) \
            + 3*self.j_pump(self.Na_si, self.K_se) \
            + self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se)
        return j
       
    def j_K_s(self, phi_sm, E_K_s):
        j = LeakyCell.j_K_s(self, phi_sm, E_K_s) \
            - 2*self.j_pump(self.Na_si, self.K_se) \
            + self.j_kcc2(self.K_si, self.K_se, self.Cl_si, self.Cl_se) \
            + self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se)
        return j

    def j_Cl_s(self, phi_sm, E_Cl_s):
        j = LeakyCell.j_Cl_s(self, phi_sm, E_Cl_s) \
            + self.j_kcc2(self.K_si, self.K_se, self.Cl_si, self.Cl_se) \
            + 2*self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se)
        return j
            
    def j_Na_d(self, phi_dm, E_Na_d):
        j = LeakyCell.j_Na_d(self, phi_dm, E_Na_d) \
            + 3*self.j_pump(self.Na_di, self.K_de) \
            + self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de)
        return j

    def j_K_d(self, phi_dm, E_K_d):
        j = LeakyCell.j_K_d(self, phi_dm, E_K_d) \
            - 2*self.j_pump(self.Na_di, self.K_de) \
            + self.j_kcc2(self.K_di, self.K_de, self.Cl_di, self.Cl_de) \
            + self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de)
        return j

    def j_Cl_d(self, phi_dm, E_Cl_d):
        j = LeakyCell.j_Cl_d(self, phi_dm, E_Cl_d) \
            + self.j_kcc2(self.K_di, self.K_de, self.Cl_di, self.Cl_de) \
            + 2*self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de)
        return j

if __name__ == "__main__":

    T = 309.14
    alpha = 2

#    Na_si0 = 18.
#    K_si0 = 140.
#    Cl_si0 = 6.
#    Ca_si0 = 0.001
#
#    Na_se0 = 144.
#    K_se0 = 4.
#    Cl_se0 = 130.
#    Ca_se0 = 1.1 
#
#    Na_di0 = 18.
#    K_di0 = 140.
#    Cl_di0 = 6.
#    Ca_di0 = 0.001
#
#    Na_de0 = 144.
#    K_de0 = 4.
#    Cl_de0 = 130.
#    Ca_de0 = 1.1

    Na_si0 = 18.
    Na_se0 = 144.
    K_si0 = 140.
    K_se0 = 4.
    Cl_si0 = 6.
    Cl_se0 = 130.
    Ca_si0 = 0.001
    Ca_se0 = 1.1

    Na_di0 = 12.
    Na_de0 = 141.
    K_di0 = 130.
    K_de0 = 3.
    Cl_di0 = 7.
    Cl_de0 = 130.
    Ca_di0 = 0.0011
    Ca_de0 = 1.1

    k_res_si = Cl_si0 - Na_si0 - K_si0 - 2*Ca_si0 #-0.035
    k_res_se = Cl_se0 - Na_se0 - K_se0 - 2*Ca_se0 #+0.07
    k_res_di = Cl_di0 - Na_di0 - K_di0 - 2*Ca_di0 #-0.035
    k_res_de = Cl_de0 - Na_de0 - K_de0 - 2*Ca_de0 #+0.07

    def dkdt(t,k):

        Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de = k

        my_cell = Pump(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha)

        dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, \
            dCadt_si, dCadt_se, dCadt_di, dCadt_de, dresdt_si, dresdt_se, dresdt_di, dresdt_de = my_cell.dkdt()

        #if t > 1 and t < 200:
        #    dKdt_si, dKdt_se = somatic_injection_current(my_cell, dKdt_si, dKdt_se, my_cell.Z_K, 500e-12)

        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dCldt_si, dCldt_se, dCldt_di, dCldt_de, \
            dCadt_si, dCadt_se, dCadt_di, dCadt_de
    
    start_time = time.time()
    t_span = (0, 2)

    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0]

    init_cell = Pump(T, Na_si0, Na_se0, Na_di0, Na_de0, K_si0, K_se0, K_di0, K_de0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, k_res_si, k_res_se, k_res_di, k_res_de, alpha)

    q_si = init_cell.total_charge([init_cell.Na_si, init_cell.K_si, init_cell.Cl_si, init_cell.Ca_si], init_cell.k_res_si, init_cell.V_si)
    q_se = init_cell.total_charge([init_cell.Na_se, init_cell.K_se, init_cell.Cl_se, init_cell.Ca_se], init_cell.k_res_se, init_cell.V_se)        
    q_di = init_cell.total_charge([init_cell.Na_di, init_cell.K_di, init_cell.Cl_di, init_cell.Ca_di], init_cell.k_res_di, init_cell.V_di)
    q_de = init_cell.total_charge([init_cell.Na_de, init_cell.K_de, init_cell.Cl_de, init_cell.Ca_de], init_cell.k_res_de, init_cell.V_de)
    print("initial total charge(C): ", q_si + q_se + q_di + q_de)
    print("Q_si (C): ", q_si)
    print("Q_se (C): ", q_se)
    print("Q_di (C): ", q_di)
    print("Q_de (C): ", q_de)

    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = init_cell.membrane_potentials()
    
    E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = init_cell.reversal_potentials()

    print("----------------------------")
    print("Initial values")
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
    print('E_Cl_d:', E_Cl_d)
    print('E_Ca_s: ', E_Ca_s)
    print('E_Ca_d:', E_Ca_d)
    print("----------------------------")

    sol = solve_ivp(dkdt, t_span, k0, max_step=1e-4)

    Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de = sol.y
    t = sol.t

    my_cell = Pump(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_di, k_res_de, alpha)
    
    phi_si, phi_se, phi_di, phi_de, phi_sm, phi_dm = my_cell.membrane_potentials()
    
    E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = my_cell.reversal_potentials()

    q_si = my_cell.total_charge([my_cell.Na_si[-1], my_cell.K_si[-1], my_cell.Cl_si[-1], my_cell.Ca_si[-1]], my_cell.k_res_si, my_cell.V_si)
    q_se = my_cell.total_charge([my_cell.Na_se[-1], my_cell.K_se[-1], my_cell.Cl_se[-1], my_cell.Ca_se[-1]], my_cell.k_res_se, my_cell.V_se)        
    q_di = my_cell.total_charge([my_cell.Na_di[-1], my_cell.K_di[-1], my_cell.Cl_di[-1], my_cell.Ca_di[-1]], my_cell.k_res_di, my_cell.V_di)
    q_de = my_cell.total_charge([my_cell.Na_de[-1], my_cell.K_de[-1], my_cell.Cl_de[-1], my_cell.Ca_de[-1]], my_cell.k_res_de, my_cell.V_de)
    print("total charge at the end (C): ", q_si + q_se + q_di + q_de)
    print("Q_si (C): ", q_si)
    print("Q_se (C): ", q_se)
    print("Q_di (C): ", q_di)
    print("Q_de (C): ", q_de)
    
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
