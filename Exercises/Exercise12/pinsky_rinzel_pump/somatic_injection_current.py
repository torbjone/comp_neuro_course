def somatic_injection_current(neuron, dkdt_si, dkdt_se, Z, I_stim):

    dkdt_si += I_stim / (neuron.V_si * neuron.F * Z)
    dkdt_se -= I_stim / (neuron.V_se * neuron.F * Z)
    
    return dkdt_si, dkdt_se
