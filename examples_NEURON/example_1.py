#!/usr/env/bin python
# -*- coding: utf-8 -*-
'''
NEURON and Python - Creating a single-compartment model with DC
current stimulus
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


##################################################################
# Model instrumentation
##################################################################

# Attach current clamp to the neuron
iclamp = neuron.h.IClamp(0.5, sec=soma)
iclamp.delay = 100. # current delay period in ms
iclamp.dur = 100.   # duration of stimulus current in ms
iclamp.amp = 0.2    # amplitude of current in nA

# print out section information again
neuron.h.psection()

##################################################################
# Set up recording of variables
##################################################################
# NEURON variables can be recorded using Vector objects. Here, we
# set up recordings of time, voltage and stimulus current with the
# record attributes.
t = neuron.h.Vector()   
v = neuron.h.Vector()
i = neuron.h.Vector()
# recordable variables must be preceded by '_ref_':
t.record(neuron.h._ref_t)   
v.record(soma(0.5)._ref_v)
i.record(iclamp._ref_i)


##################################################################
# Simulation control
##################################################################
neuron.h.dt = 0.1          # simulation time resolution
tstop = 300.        # simulation duration
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
fig.suptitle('stimulus current and point-neuron response')
axes[0].plot(t, i, 'r', lw=2)
axes[0].set_ylabel('current (nA)')

axes[1].plot(t, v, 'r', lw=2)
axes[1].set_ylabel('voltage (mV)')
axes[1].set_xlabel('time (ms)')

# tight layout
for ax in axes: ax.axis(ax.axis('tight'))

fig.savefig('example_1.pdf')
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
iclamp = None
soma = None
