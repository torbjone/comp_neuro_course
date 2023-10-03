#!/usr/env/bin python
# -*- coding: utf-8 -*-
'''
NEURON and Python - Creating a single-compartment model
with synaptic input
'''
# Import modules for plotting and NEURON itself 
import matplotlib.pyplot as plt
import neuron

################################################################################
# Neuron topology is defined using Sections
################################################################################
soma = neuron.h.Section(name='soma')    # soma section

#print out information on the different sections to stdout (terminal)
neuron.h.psection()


################################################################################
# Set the model geometry
################################################################################
soma.L = 30.        # section length in um
soma.diam = 30.     # section diameter in um
soma.nseg = 1       # number of segments (compartments)


################################################################################
# Set biophysical parameters
################################################################################
soma.Ra = 100       # Axial resistivity in Ohm*cm
soma.cm = 1         # membrane capacitance in uF/cm2

# insert 'passive' membrane mechanism, adjust parameters.
# None: without a leak mechanism, the neuron will be a
# perfect integrator
soma.insert('pas')          
soma(0.5).pas.g = 0.0002    # membrane conducance in S/cm2
soma(0.5).pas.e = -65.       # leak reversal potential in mV


################################################################################
# Model instrumentation
################################################################################

# Connect synapse objects to soma object
syn0 = neuron.h.AlphaSynapse(0.5, sec=soma)
syn0.onset = 100.   # onset time of synapse in ms
syn0.gmax = 0.01    # maximum conductance of synapse in uS
syn0.e = 0.         # reversal potential of synapse conductance in mV
syn0.tau = 2.       # time constant of synapse conductance in ms

# 'inhibitory' synapse
syn1 = neuron.h.AlphaSynapse(0.5, sec=soma)
syn1.onset = 200.   # onset time of synapse in ms
syn1.gmax = 0.01    # maximum conductance of synapse in uS
syn1.e = -80.       # reversal potential of synapse conductance in mV
syn1.tau = 2.       # time constant of synapse conductance in ms

# print out section information again
neuron.h.psection()

################################################################################
# Set up recording of variables
################################################################################
t = neuron.h.Vector()   # NEURON variables can be recorded using Vector objects.
v = neuron.h.Vector()   # Here, we set up recordings of time, voltage
isyn0 = neuron.h.Vector()   # and synapse currents with the record attributes. 
isyn1 = neuron.h.Vector()

t.record(neuron.h._ref_t)
v.record(soma(0.5)._ref_v)
isyn0.record(syn0._ref_i)
isyn1.record(syn1._ref_i)


################################################################################
# Simulation control
################################################################################
neuron.h.dt = 0.1          # simulation time resolution
tstop = 300.        # simulation duration
v_init = -65        # membrane voltage(s) at t = 0

def initialize():
    '''
    initializing function, setting the membrane voltages to v_init and
    resetting all state variables
    '''
    neuron.h.finitialize(v_init)
    neuron.h.fcurrent()

def integrate():
    '''
    run the simulation up until the simulation duration
    '''
    while neuron.h.t < tstop:
        neuron.h.fadvance()

# run simulation
initialize()
integrate()


################################################################################
# Plot simulated output
################################################################################
fig, axes = plt.subplots(2)
fig.suptitle('synapse currents and neuron response')
axes[0].plot(t, isyn0, 'r', lw=2)
axes[0].plot(t, isyn1, 'b', lw=2)
axes[0].set_ylabel('current (nA)')

axes[1].plot(t, v, 'r', lw=2)
axes[1].set_ylabel('voltage (mV)')
axes[1].set_xlabel('time (ms)')

for ax in axes: ax.axis(ax.axis('tight'))

fig.savefig('example_3.pdf')
plt.show()


################################################################################
# customary cleanup of object references - the psection() function may not write
# correct information if NEURON still has object references in memory, even if
# Python references has been deleted.
################################################################################
plt.close(fig)
v = None
t = None
seg = None
sec = None
isyn0 = None
isyn1 = None
syn0 = None
syn1 = None
axon = None
dend = None
soma = None
apic = None
