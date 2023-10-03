#!/usr/env/bin python
# -*- coding: utf-8 -*-
'''
NEURON and Python - Creating a single-compartment model with
synapse randomized activation times
'''
# Import modules for plotting and NEURON itself 
import matplotlib.pyplot as plt
import neuron

##################################################################
# Neuron topology is defined using Sections
##################################################################
soma = neuron.h.Section(name='soma')

#print out information on the soma section to terminal
neuron.h.psection()


##################################################################
# Set the model geometry
##################################################################
soma.L = 30         # section length in um
soma.diam = 30      # section diameter in um
soma.nseg = 1       # number of segments (compartments)


##################################################################
# Set biophysical parameters
##################################################################
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

# Connect synapse object to the center of the soma, here an
# 'excitatory' synapse
syn = neuron.h.ExpSyn(0.5, sec=soma)
syn.e = 0.         # reversal potential of synapse conductance in mV
syn.tau = 2.       # time constant of synapse conductance in ms


################################################################################
# create generators and connections for synapse activation times
################################################################################
ns = neuron.h.NetStim(0.5)  # spike time generator object (~presynaptic)
ns.noise = 1.               # Fractional randomness (intervals from exp dist)
ns.start = 0.               # approximate time of first spike
ns.number = 1000            # number of spikes
ns.interval = 10.           # average interspike interval
nc = neuron.h.NetCon(ns, syn)  # Connect generator to synapse
nc.weight[0] = 0.01            # Set synapse weight


# print out section information again
neuron.h.psection()

################################################################################
# Set up recording of variables
################################################################################
t = neuron.h.Vector()   # NEURON variables can be recorded using Vector objects.
v = neuron.h.Vector()   # Here, we set up recordings of time, voltage
isyn = neuron.h.Vector()   # and synapse currents with the record attributes. 

t.record(neuron.h._ref_t)
v.record(soma(0.5)._ref_v)
isyn.record(syn._ref_i)



##################################################################
# Simulation control
##################################################################
neuron.h.dt = 0.1   # simulation time resolution
tstop = 500.        # simulation duration
v_init = -65        # membrane voltage(s) at t = 0

def initialize():
    '''
    initializing function, setting the membrane voltages to v_init
    and resetting all state variables
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


##################################################################
# Plot simulated output
##################################################################
fig, axes = plt.subplots(2)
fig.suptitle('synapse current and point-neuron response')
axes[0].plot(t, isyn, 'r', lw=2)
axes[0].set_ylabel('current (nA)')

axes[1].plot(t, v, 'r', lw=2)
axes[1].set_ylabel('voltage (mV)')
axes[1].set_xlabel('time (ms)')

# tight layout
for ax in axes: ax.axis(ax.axis('tight'))

fig.savefig('example_4.pdf')
plt.show()


##################################################################
# customary cleanup of object references - the psection() function
# may not write correct information if NEURON still has object
# references in memory.
##################################################################
plt.close(fig)
i = None
v = None
t = None
ns = None
nc = None
isyn = None
soma = None
