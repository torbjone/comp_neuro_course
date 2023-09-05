# -*- coding: utf-8 -*-
#
# brunel_delta_nest.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
Random balanced network (delta synapses)
----------------------------------------

This script simulates an excitatory and an inhibitory population on
the basis of the network used in [1]_

When connecting the network, customary synapse models are used, which
allow for querying the number of created synapses. Using spike
recorders, the average firing rates of the neurons in the populations
are established. The building as well as the simulation time of the
network are recorded.

References
~~~~~~~~~~

.. [1] Brunel N (2000). Dynamics of sparsely connected networks of excitatory and
       inhibitory spiking neurons. Journal of Computational Neuroscience 8,
       183-208.

"""

###############################################################################
# Import all necessary modules for simulation, analysis and plotting.

import time
import nest
import nest.raster_plot
import matplotlib.pyplot as plt
import pandas as pd

def sim_brunel_delta(dt=0.1,
                     simtime=1000.0,
                     delay=1.5,
                     g=5.0,
                     eta=2.0,
                     epsilon=0.1,
                     order=2500,
                     J=0.1,
                     N_rec=50,
                     num_vp=1,
                     seed=122131234,
                     print_report=True,
                     V_reset=0.0,
                     input_stop=None):
                     
    nest.ResetKernel()
    nest.set_verbosity('M_WARNING')

    ###############################################################################
    # Assigning the current time to a variable in order to determine the build
    # time of the network.

    startbuild = time.time()

    NE = 4 * order  # number of excitatory neurons
    NI = 1 * order  # number of inhibitory neurons
    N_neurons = NE + NI  # number of neurons in total

    ###############################################################################
    # Definition of connectivity parameters

    CE = int(epsilon * NE)  # number of excitatory synapses per neuron
    CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
    C_tot = int(CI + CE)  # total number of synapses per neuron

    ###############################################################################
    # Initialization of the parameters of the integrate and fire neuron and the
    # synapses. The parameters of the neuron are stored in a dictionary.

    tauMem = 20.0  # time constant of membrane potential in ms
    theta = 20.0  # membrane threshold potential in mV
    neuron_params = {"C_m": 1.0,
	             "tau_m": tauMem,
	             "t_ref": 2.0,
	             "E_L": 0.0,
	             "V_reset": V_reset,
	             "V_m": 0, #nest.random.uniform(-theta, theta),
	             "V_th": theta}

    J_ex = J  # amplitude of excitatory postsynaptic potential
    J_in = -g * J_ex  # amplitude of inhibitory postsynaptic potential

    ###############################################################################
    # Definition of threshold rate, which is the external rate needed to fix the
    # membrane potential around its threshold, the external firing rate and the
    # rate of the poisson generator which is multiplied by the in-degree CE and
    # converted to Hz by multiplication by 1000.

    nu_th = theta / (J * CE * tauMem)
    nu_ex = eta * nu_th
    p_rate = 1000.0 * nu_ex * CE

    ###############################################################################
    # Configuration of the simulation kernel by the previously defined time
    # resolution used in the simulation. Setting ``print_time`` to `True` prints the
    # already processed simulation time as well as its percentage of the total
    # simulation time.

    nest.resolution = dt
    nest.print_time = True
    nest.overwrite_files = True

    nest.rng_seed = seed
    nest.total_num_virtual_procs = num_vp
    if print_report:
        print("Building network")

    ###############################################################################
    # Creation of the nodes using ``Create``. We store the returned handles in
    # variables for later reference. Here the excitatory and inhibitory, as well
    # as the poisson generator and two spike recorders. The spike recorders will
    # later be used to record excitatory and inhibitory spikes. Properties of the
    # nodes are specified via ``params``, which expects a dictionary.

    nodes_ex = nest.Create("iaf_psc_delta", NE, params=neuron_params)
    nodes_in = nest.Create("iaf_psc_delta", NI, params=neuron_params)
    if input_stop is None:
        noise = nest.Create("poisson_generator", params={"rate": p_rate})
    else:
        noise = nest.Create("poisson_generator", params={"rate": p_rate, "stop": input_stop})    
    
    espikes = nest.Create("spike_recorder")
    ispikes = nest.Create("spike_recorder")

    if print_report:
        print("Connecting devices")

    ###############################################################################
    # Definition of a synapse using ``CopyModel``, which expects the model name of
    # a pre-defined synapse, the name of the customary synapse and an optional
    # parameter dictionary. The parameters defined in the dictionary will be the
    # default parameter for the customary synapse. Here we define one synapse for
    # the excitatory and one for the inhibitory connections giving the
    # previously defined weights and equal delays.

    nest.CopyModel("static_synapse", "excitatory",
	           {"weight": J_ex, "delay": delay})
    nest.CopyModel("static_synapse", "inhibitory",
	           {"weight": J_in, "delay": delay})

    ###############################################################################
    # Connecting the previously defined poisson generator to the excitatory and
    # inhibitory neurons using the excitatory synapse. Since the poisson
    # generator is connected to all neurons in the population the default rule
    # (# ``all_to_all``) of ``Connect`` is used. The synaptic properties are inserted
    # via ``syn_spec`` which expects a dictionary when defining multiple variables
    # or a string when simply using a pre-defined synapse.

    nest.Connect(noise, nodes_ex, syn_spec="excitatory")
    nest.Connect(noise, nodes_in, syn_spec="excitatory")

    ###############################################################################
    # Connecting the first ``N_rec`` nodes of the excitatory and inhibitory
    # population to the associated spike recorders using excitatory synapses.
    # Here the same shortcut for the specification of the synapse as defined
    # above is used.

    nest.Connect(nodes_ex[:N_rec], espikes, syn_spec="excitatory")
    nest.Connect(nodes_in[:N_rec], ispikes, syn_spec="excitatory")

    if print_report:
        print("Connecting network")
        print("Excitatory connections")

    ###############################################################################
    # Connecting the excitatory population to all neurons using the pre-defined
    # excitatory synapse. Beforehand, the connection parameter are defined in a
    # dictionary. Here we use the connection rule ``fixed_indegree``,
    # which requires the definition of the indegree. Since the synapse
    # specification is reduced to assigning the pre-defined excitatory synapse it
    # suffices to insert a string.

    conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}
    nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, "excitatory")

    if print_report:
        print("Inhibitory connections")

    ###############################################################################
    # Connecting the inhibitory population to all neurons using the pre-defined
    # inhibitory synapse. The connection parameters as well as the synapse
    # parameters are defined analogously to the connection from the excitatory
    # population defined above.

    conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
    nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, "inhibitory")

    ###############################################################################
    # Storage of the time point after the buildup of the network in a variable.

    endbuild = time.time()

    ###############################################################################
    # Simulation of the network.

    if print_report:
        print("Simulating")

    nest.Simulate(simtime)

    ###############################################################################
    # Storage of the time point after the simulation of the network in a variable.

    endsimulate = time.time()

    ###############################################################################
    # Reading out the total number of spikes received from the spike recorder
    # connected to the excitatory population and the inhibitory population.

    events_ex = espikes.n_events
    events_in = ispikes.n_events

    ###############################################################################
    # Calculation of the average firing rate of the excitatory and the inhibitory
    # neurons by dividing the total number of recorded spikes by the number of
    # neurons recorded from and the simulation time. The multiplication by 1000.0
    # converts the unit 1/ms to 1/s=Hz.

    rate_ex = events_ex / simtime * 1000 / N_rec
    rate_in = events_in / simtime * 1000 / N_rec

    ###############################################################################
    # Reading out the number of connections established using the excitatory and
    # inhibitory synapse model. The numbers are summed up resulting in the total
    # number of synapses.

    num_synapses = (nest.GetDefaults("excitatory")["num_connections"] +
	            nest.GetDefaults("inhibitory")["num_connections"])

    ###############################################################################
    # Establishing the time it took to build and simulate the network by taking
    # the difference of the pre-defined time variables.

    build_time = endbuild - startbuild
    sim_time = endsimulate - endbuild

    ###############################################################################
    # Printing the network properties, firing rates and building times.
    if print_report:
        print("Brunel network simulation (Python)")
        print(f"Number of neurons : {N_neurons}")
        print(f"Number of synapses: {num_synapses}")
        print(f"       Exitatory  : {int(CE * N_neurons) + N_neurons}")
        print(f"       Inhibitory : {int(CI * N_neurons)}")
        print(f"Excitatory rate   : {rate_ex:.2f} Hz")
        print(f"Inhibitory rate   : {rate_in:.2f} Hz")
        print(f"Building time     : {build_time:.2f} s")
        print(f"Simulation time   : {sim_time:.2f} s")


    exc_spikes = espikes.get('events')
    inh_spikes = ispikes.get('events')

    return pd.DataFrame(exc_spikes), pd.DataFrame(inh_spikes)



