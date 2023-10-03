#!/usr/env/bin python
# -*- coding: utf-8 -*-
'''
NEURON and Python - Creating a single-compartment model with noisy current
input
'''
# Import modules for plotting and NEURON itself 
import matplotlib.pyplot as plt
import numpy as np
import neuron

# Fix seed for numpy random number generation
np.random.seed(1234)

################################################################################
# Neuron topology is defined using Sections
################################################################################
soma = neuron.h.Section(name='soma')

#print out information on the soma section to stdout (terminal)
neuron.h.psection()


################################################################################
# Set the model geometry
################################################################################
soma.L = 30         # section length in um
soma.diam = 30      # section diameter in um
soma.nseg = 1       # number of segments (compartments)


################################################################################
# Set biophysical parameters
################################################################################
soma.Ra = 100       # Axial resistivity in Ohm*cm
soma.cm = 1         # membrane capacitance in uF/cm2

#insert Hodkin-Huxley type membrane mechanism in the soma
soma.insert('hh')          
# similarly one can adjust the parameters for the 'hh' mechanism
# but we will not do that here.
# soma(0.5).hh.gnabar_hh = 0.12
# soma(0.5).hh.gkbar_hh = 0.036
# soma(0.5).hh.gl_hh = 0.0003
# soma(0.5).hh.el_hh = -54.3
# soma(0.5).na_ion.ena = 50.
# soma(0.5).k_ion.ek = -77.

################################################################################
# Model instrumentation
################################################################################

# Attach current clamp to the neuron
iclamp = neuron.h.IClamp(0.5, sec=soma)
iclamp.delay = 0. # switch on current after delay period in ms
iclamp.dur = 1E9   # duration of stimulus current in ms

# create a white noise signal with values from a Gaussian distribution with
# expected mean of zero and standard deviation of 0.25. 
noise = neuron.h.Vector(np.random.randn(10000) / 4.)

# Play back noise signal into clamp amplitude reference with update every dt.
dt = 0.1
noise.play(iclamp._ref_amp, dt)

# print out section information again
neuron.h.psection()

################################################################################
# Set up recording of variables
################################################################################
t = neuron.h.Vector()   # NEURON variables can be recorded using Vector objects.
v = neuron.h.Vector()   # Here, we set up recordings of time, voltage
i = neuron.h.Vector()   # and stimulus current with the record attributes. 

t.record(neuron.h._ref_t)   # recordable variables must be preceded by '_ref_'.
v.record(soma(0.5)._ref_v)
i.record(iclamp._ref_i)


################################################################################
# Simulation control
################################################################################
neuron.h.dt = dt    # simulation time resolution
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
fig.suptitle('stimulus current and neuron response')
axes[0].plot(t, i, 'r', lw=2)
axes[0].set_ylabel('current (nA)')

axes[1].plot(t, v, 'r', lw=2)
axes[1].set_ylabel('voltage (mV)')
axes[1].set_xlabel('time (ms)')

for ax in axes: ax.axis(ax.axis('tight'))

fig.savefig('example_5.pdf')
plt.show()
plt.close(fig)


################################################################################
# customary cleanup of object references - the psection() function may not write
# correct information if NEURON still has object references in memory.
################################################################################
i = None
v = None
t = None
iclamp = None
soma = None
