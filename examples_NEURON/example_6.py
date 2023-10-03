#!/usr/env/bin python
# -*- coding: utf-8 -*-
'''
NEURON and Python - Creating a multi-compartment model with synaptic input
with randomized activation times

'''
# Import modules for plotting and NEURON itself 
import matplotlib.pyplot as plt
import neuron

################################################################################
# Neuron topology is defined using Sections
################################################################################
soma = neuron.h.Section(name='soma')    # soma section
apic = neuron.h.Section(name='apic')    # apical dendrite representation
dend = neuron.h.Section(name='dend')    # basal dendrites
axon = neuron.h.Section(name='axon')    # axon section

# Connect the different section elements
apic.connect(soma(1), 0)    # connect 'start' of apical dendrite soma 'end'
dend.connect(soma(0), 0)    # connect 'start' of basal dendrite to soma 'start'
axon.connect(soma(0), 0)    # ditto for axon 

#print out information on the different sections to stdout (terminal)
for sec in neuron.h.allsec():
    neuron.h.psection(sec=sec)


################################################################################
# Set the model geometry
################################################################################
soma.L = 30.        # section length in um
soma.diam = 30.     # section diameter in um
soma.nseg = 1       # number of segments (compartments)

apic.L = 600.
apic.diam = 1.
apic.nseg = 25      # apical dendrite is now discretized in 25 segments

dend.L = 200.
dend.diam = 2.
dend.nseg = 5

axon.L = 1000.
axon.diam = 1.
axon.nseg = 35

# print out the model topology to stdout:
neuron.h.topology()

################################################################################
# Set biophysical parameters
################################################################################
# set up global axial resitivity and membrane capacitance
for sec in neuron.h.allsec():
    sec.Ra = 100       # Axial resistivity in Ohm*cm
    sec.cm = 1         # membrane capacitance in uF/cm2

# insert 'passive' membrane mechanism in dendrites only, adjust parameters
for sec in [apic, dend]:
    sec.insert('pas')
    for seg in sec:
        seg.pas.g = 0.0002    # membrane conductance in S/cm2
        seg.pas.e = -65.       # passive leak reversal potential in mV
    
#insert Hodkin-Huxley type membrane mechanism in the soma and axonal sections
for sec in [soma, axon]:
    sec.insert('hh')


################################################################################
# Model instrumentation
################################################################################

# Connect synapse object to the center of the apical dendrite, here an
# 'excitatory' synapse
syn = neuron.h.ExpSyn(0.5, sec=apic)
syn.e = 0.         # reversal potential of synapse conductance in mV
syn.tau = 2.       # time constant of synapse conductance in ms


################################################################################
# create generators and connections for synapse activation times
################################################################################
ns = neuron.h.NetStim(0.5)     # spike time generator object (~presynaptic)
ns.noise = 1.                  # Fractional randomness (intervals from exp dist)
ns.start = 0.                  # approximate time of first spike
ns.number = 1000               # number of spikes
ns.interval = 10.              # average interspike interval
nc = neuron.h.NetCon(ns, syn)    # Connect generator to synapse
nc.weight[0] = 2.                  # Set synapse weight


# print out section information again
for sec in neuron.h.allsec():
    neuron.h.psection(sec=sec)

################################################################################
# Set up recording of variables
################################################################################
t = neuron.h.Vector()   # NEURON variables can be recorded using Vector objects.
v = neuron.h.Vector()   # Here, we set up recordings of time, voltage
isyn = neuron.h.Vector()   # and synapse currents with the record attributes. 

t.record(neuron.h._ref_t)
v.record(soma(0.5)._ref_v)
isyn.record(syn._ref_i)


################################################################################
# Simulation control
################################################################################
neuron.h.dt = 0.1          # simulation time resolution
tstop = 500.        # simulation duration
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
fig.suptitle('synapse current and neuron response')
axes[0].plot(t, isyn, 'r', lw=2)
axes[0].set_ylabel('current (nA)')

axes[1].plot(t, v, 'r', lw=2)
axes[1].set_ylabel('voltage (mV)')
axes[1].set_xlabel('time (ms)')

for ax in axes: ax.axis(ax.axis('tight'))

fig.savefig('example_6.pdf')
plt.show()


################################################################################
# customary cleanup of object references - the psection() function may not write
# correct information if NEURON still has object references in memory, even if
# Python references has been deleted.
################################################################################
plt.close(fig)
v = None
t = None
nc = None
ns = None
seg = None
sec = None
isyn = None
syn = None
axon = None
dend = None
soma = None
apic = None
