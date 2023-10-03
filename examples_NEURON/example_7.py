#!/usr/env/bin python
# -*- coding: utf-8 -*-
'''
NEURON and Python - Create a small network of mutually connected
single-compartment neurons with Hodkin-Huxley style membrane properties

'''
# Import modules for plotting and NEURON itself 
import matplotlib.pyplot as plt
import neuron

################################################################################
# Create a class that can be reused for multiple neuron representations
################################################################################
class HHCell(neuron.nrn.Section):
    '''
    Cell class based on inheritance from NEURON's Section object, that allows
    for setting parameters upon creation. The Hodkin-Huxley formalism is
    inserted into the cell by default
    '''
    def __init__(self, L=30., diam=30., Ra=100., cm=1.,):
        '''
        Parameters
        ----------
        L : float, section length
        diam : float, section diameter
        Ra : float, axial resistivity
        cm : float, membrane capacitance
        
        '''
        neuron.nrn.Section.__init__(self)
        # Set Section attributes
        self.L = L
        self.diam = diam
        self.Ra = Ra
        self.cm = cm
        
        # Insert Hodkin-Huxley formalism
        self.insert('hh')

        #create recording device for membrane voltage
        self.vm = neuron.h.Vector()
        self.vm.record(self(0.5)._ref_v)
        
        # create lists containing Synapses, NetStim and NetCon devices
        self.synapses = []
        self.netstims = []
        self.netcons = []


    def create_noise_input(self, noise=1., start=0., number=10000, interval=10.,
                           tau=1., e=0., weight=0.002):
        '''
        Create and attach noisy input to neuron.
        
        Parameters
        ----------
        noise : float, Fractional randomness (1 = intervals from exp dist)
        start : float, approximate time of first spike
        number : int, number of delivered spikes
        interval : float, average interspike interval
        tau : float, synapse time constant
        e : float, synapse reversal potential
        weight : float, synapse strength
        '''
        # Create a NetStim device to activate synapse
        self.netstims.append(neuron.h.NetStim(0.5))    # spike generator object
        self.netstims[-1].noise = noise
        self.netstims[-1].start = start
        self.netstims[-1].number = number
        self.netstims[-1].interval = interval
        # Some may wonder about indexing the last element, turns out one cannot
        # create python object, set the parameters and then append the object to
        # the list. The network may break when the original python object
        # reference is destroyed, i.e., when the function is done executing. 
        
        # Create a synapse
        self.synapses.append(neuron.h.ExpSyn(0.5, sec=self))
        self.synapses[-1].tau = tau
        self.synapses[-1].e = e
        
        # Connect NetStim device to synapse
        self.netcons.append(neuron.h.NetCon(self.netstims[-1],
                                            self.synapses[-1]))
        self.netcons[-1].weight[0] = weight
        

    def connect_cell(self, target, threshold=0., tau=2., e=-80.,
                     weight=0.1, delay=2.):
        '''
        Connect synapse on postsynaptic cell to this cell with a specific
        weight and delay, triggered on this presynaptic cell's potential
        
        Parameters
        ----------
        target : HHCell object, presynaptic neuron
        threshold : float, spike detection threshold
        tau : float, synapse time constant
        e : float, synapse reversal potential
        weight : float, connection weight
        delay : float, connection delay
        '''
        # Create a synapse object on the target cell
        target.synapses.append(neuron.h.ExpSyn(0.5, sec=target))
        target.synapses[-1].tau = tau
        target.synapses[-1].e = e

        
        # Create new NetCon objects for the connections
        self.netcons.append(neuron.h.NetCon(self(0.5)._ref_v,
                                            target.synapses[-1],
                                            sec=self))
        self.netcons[-1].threshold = threshold
        self.netcons[-1].weight[0] = weight
        self.netcons[-1].delay = delay

 
################################################################################
# fill in a list with HHCell objects
################################################################################
cells = [HHCell() for x in range(5)]

################################################################################
# create noisy input to each cell to keep them activated
################################################################################
for i in range(len(cells)):
    cells[i].create_noise_input()

################################################################################
# connect all cells to all other cells in an inhibitory network
# (avoiding connections to self)
################################################################################
for i in range(len(cells)):     # pre
    for j in range(len(cells)): # post
        if i != j:
            cells[i].connect_cell(cells[j])

################################################################################
# Recording of additional variables
################################################################################
t = neuron.h.Vector()
t.record(neuron.h._ref_t)


################################################################################
# Simulation control
################################################################################
neuron.h.dt = 0.1   # simulation time resolution
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
fig, axes = plt.subplots(len(cells))
fig.suptitle('point-neuron responses')
for i, cell in enumerate(cells):
    axes[i].plot(t, cell.vm, 'r', lw=2)
    axes[i].axis(axes[i].axis('tight'))
    axes[i].set_ylabel('cell {}'.format(i+1))
axes[i].set_xlabel('time (ms)')

fig.savefig('example_7.pdf')
plt.show()
plt.close(fig)


################################################################################
# customary cleanup of object references - the psection() function may not write
# correct information if NEURON still has object references in memory, even if
# Python references has been deleted.
################################################################################
for cell in cells:
    cell = None
cells = None

