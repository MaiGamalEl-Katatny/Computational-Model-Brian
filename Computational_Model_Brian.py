# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 12:52:13 2020

@author: maiga
"""

#This project is based on the model presented in: Ramirez-Mahaluf, Juan P., et al. "A computational model of Major Depression: the role of glutamate dysfunction on cingulo-frontal network dynamics." Cerebral cortex 27.1 (2017): 660-679.

import brian2
import timeit
from brian2 import *
import matplotlib
import matplotlib.pyplot as plt

#Start Computation Time Calculation
start = timeit.default_timer()

#Network Simulation function
def simulate(tau_AMPA_vACC, V_L_T, g_NMDA_E_vACC):

    start_scope()
    
    # populations
    N = 1000
    N_E = int(N * 0.8)  # pyramidal neurons
    N_I = int(N * 0.2)  # interneurons
    
    # voltage
    V_L = -70. * mV
    V_thr = -50. * mV
    V_reset = -55. * mV
    V_E = 0. * mV
    V_I = -70. * mV
    
    # membrane capacitance
    C_m_E = 0.5 * nF
    C_m_I = 0.2 * nF
    
    # membrane leak
    g_m_E = 25. * nS
    g_m_I = 20. * nS
    
    # refractory period
    tau_rp_E = 2. * ms
    tau_rp_I = 1. * ms
    
    # external stimuli
    rate = 1800 * Hz
    C_ext = 1
    C_inh = 1
    
    # synapses
    C_E = N_E
    C_I = N_I
    
    # AMPA (excitatory)
    g_AMPA_ext_E = 0.21 * nS
    g_AMPA_rec_E = 0.024* nS * 800. / N_E
    g_AMPA_ext_I = 0.16 * nS
    g_AMPA_rec_I = 0.008 * nS * 800. / N_E
    #tau_AMPA_vACC = 2. * ms
    tau_AMPA_dlPFC = 2. * ms
    
    # NMDA (excitatory)
    g_NMDA_E = 0.044 * nS * 800. / N_E
    g_NMDA_I = 0.024 * nS * 800. / N_E
    tau_NMDA_rise = 2. * ms
    tau_NMDA_decay = 100. * ms
    alpha = 0.5 / ms
    Mg2 = 1.
    
    # GABAergic (inhibitory)
    g_GABA_E = 0.1 * nS * 200. / N_I
    g_GABA_I = 0.097 * nS * 200. / N_I
    tau_GABA = 10. * ms
    
    # modeling
    eqs_E_vACC = '''
    dv / dt = (- g_m_E * (v - V_L_T) - I_syn) / C_m_E : volt (unless refractory)
    
    I_syn = I_AMPA_ext + I_AMPA_rec + I_NMDA_rec + I_GABA_rec : amp
    
    I_AMPA_ext = g_AMPA_ext_E * (v - V_E) * s_AMPA_ext : amp
    I_AMPA_rec = g_AMPA_rec_E * (v - V_E) * 1 * s_AMPA : amp
    ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA_vACC : 1
    ds_AMPA / dt = - s_AMPA / tau_AMPA_vACC : 1
    
    I_NMDA_rec = g_NMDA_E_vACC * (v - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
    s_NMDA_tot : 1
    
    I_GABA_rec = g_GABA_E * (v - V_I) * s_GABA : amp
    ds_GABA / dt = - s_GABA / tau_GABA : 1
    '''
    
    eqs_I_vACC = '''
    dv / dt = (- g_m_I * (v - V_L) - I_syn) / C_m_I : volt (unless refractory)
    
    I_syn = I_AMPA_ext + I_AMPA_rec + I_NMDA_rec + I_GABA_rec : amp
    
    I_AMPA_ext = g_AMPA_ext_I * (v - V_E) * s_AMPA_ext : amp
    I_AMPA_rec = g_AMPA_rec_I * (v - V_E) * 1 * s_AMPA  : amp
    ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA_vACC : 1
    ds_AMPA / dt = - s_AMPA / tau_AMPA_vACC : 1
    
    I_NMDA_rec = g_NMDA_I * (v - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
    s_NMDA_tot : 1
    
    I_GABA_rec = g_GABA_I * (v - V_I) * s_GABA : amp
    ds_GABA / dt = - s_GABA / tau_GABA : 1
    '''
    
    eqs_E_dlPFC = '''
    dv / dt = (- g_m_E * (v - V_L) - I_syn) / C_m_E : volt (unless refractory)
    
    I_syn = I_AMPA_ext + I_AMPA_rec + I_NMDA_rec + I_GABA_rec : amp
    
    I_AMPA_ext = g_AMPA_ext_E * (v - V_E) * s_AMPA_ext : amp
    I_AMPA_rec = g_AMPA_rec_E * (v - V_E) * 1 * s_AMPA : amp
    ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA_dlPFC : 1
    ds_AMPA / dt = - s_AMPA / tau_AMPA_dlPFC : 1
    
    I_NMDA_rec = g_NMDA_E * (v - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
    s_NMDA_tot : 1
    
    I_GABA_rec = g_GABA_E * (v - V_I) * s_GABA : amp
    ds_GABA / dt = - s_GABA / tau_GABA : 1
    '''
    
    eqs_I_dlPFC = '''
    dv / dt = (- g_m_I * (v - V_L) - I_syn) / C_m_I : volt (unless refractory)
    
    I_syn = I_AMPA_ext + I_AMPA_rec + I_NMDA_rec + I_GABA_rec : amp
    
    I_AMPA_ext = g_AMPA_ext_I * (v - V_E) * s_AMPA_ext : amp
    I_AMPA_rec = g_AMPA_rec_I * (v - V_E) * 1 * s_AMPA  : amp
    ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA_dlPFC : 1
    ds_AMPA / dt = - s_AMPA / tau_AMPA_dlPFC : 1
    
    I_NMDA_rec = g_NMDA_I * (v - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
    s_NMDA_tot : 1
    
    I_GABA_rec = g_GABA_I * (v - V_I) * s_GABA : amp
    ds_GABA / dt = - s_GABA / tau_GABA : 1
    '''
    
    P_E_vACC = NeuronGroup(N_E, eqs_E_vACC, threshold='v > V_thr', reset='v = V_reset', refractory=tau_rp_E, method='euler')
    P_E_vACC.v = V_L
    P_I_vACC = NeuronGroup(N_I, eqs_I_vACC, threshold='v > V_thr', reset='v = V_reset', refractory=tau_rp_I, method='euler')
    P_I_vACC.v = V_L
    P_E_dlPFC = NeuronGroup(N_E, eqs_E_dlPFC, threshold='v > V_thr', reset='v = V_reset', refractory=tau_rp_E, method='euler')
    P_E_dlPFC.v = V_L
    P_I_dlPFC = NeuronGroup(N_I, eqs_I_dlPFC, threshold='v > V_thr', reset='v = V_reset', refractory=tau_rp_I, method='euler')
    P_I_dlPFC.v = V_L
    
    
    eqs_glut = '''
    s_NMDA_tot_post = s_NMDA : 1 (summed)
    ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha * x * (1 - s_NMDA) : 1 (clock-driven)
    dx / dt = - x / tau_NMDA_rise : 1 (clock-driven)
    '''
    
    eqs_pre_glut = '''
    s_AMPA_post += 1
    x += 1
    '''
    
    eqs_pre_ampa = '''
    s_AMPA_post += w
    '''
    
    eqs_pre_gaba = '''
    s_GABA_post += 1
    '''
    
    eqs_pre_ext = '''
    s_AMPA_ext_post += 1
    '''
    
    # E to E NMDA
    C_E_E_vACC = Synapses(P_E_vACC, P_E_vACC, model=eqs_glut, on_pre=eqs_pre_glut, method='euler')
    C_E_E_vACC.connect('i != j')
    
    # E to I
    C_E_I_vACC = Synapses(P_E_vACC, P_I_vACC, model=eqs_glut, on_pre=eqs_pre_glut, method='euler')
    C_E_I_vACC.connect()
    
    # I to I
    C_I_I_vACC = Synapses(P_I_vACC, P_I_vACC, on_pre=eqs_pre_gaba, method='euler')
    C_I_I_vACC.connect('i != j')
    
    # I to E
    C_I_E_vACC = Synapses(P_I_vACC, P_E_vACC, on_pre=eqs_pre_gaba, method='euler')
    C_I_E_vACC.connect()
    
    
    # E to E NMDA
    C_E_E_dlPFC = Synapses(P_E_dlPFC, P_E_dlPFC, model = eqs_glut, on_pre=eqs_pre_glut, method='euler')
    C_E_E_dlPFC.connect('i != j')
    
    # E to I
    C_E_I_dlPFC = Synapses(P_E_dlPFC, P_I_dlPFC, model=eqs_glut, on_pre=eqs_pre_glut, method='euler')
    C_E_I_dlPFC.connect()
    
    # I to I
    C_I_I_dlPFC = Synapses(P_I_dlPFC, P_I_dlPFC, on_pre=eqs_pre_gaba, method='euler')
    C_I_I_dlPFC.connect('i != j')
    
    # I to E
    C_I_E_dlPFC = Synapses(P_I_dlPFC, P_E_dlPFC, on_pre=eqs_pre_gaba, method='euler')
    C_I_E_dlPFC.connect()
    
    #E vACC to I dlPFC
    C_v_d = Synapses(P_E_vACC, P_I_dlPFC, 'w : 1', on_pre=eqs_pre_ampa, method='euler')
    C_v_d.connect()
    C_v_d.w[:] = 1
    
    #E dlPFC to I vACC
    C_d_v = Synapses(P_E_dlPFC, P_I_vACC, 'w : 1', on_pre=eqs_pre_ampa, method='euler')
    C_d_v.connect()
    C_d_v.w[:] = 1
    
    # external noise
    C_P_E_vACC = PoissonInput(P_E_vACC, 's_AMPA_ext', C_ext, rate, '10')
    C_P_I_vACC = PoissonInput(P_I_vACC, 's_AMPA_ext', C_inh, rate, '10')
    C_P_E_dlPFC = PoissonInput(P_E_dlPFC, 's_AMPA_ext', C_ext, rate, '10')
    C_P_I_dlPFC = PoissonInput(P_I_dlPFC, 's_AMPA_ext', C_inh, rate, '10')
    
    #Simulation Tasks
    
    #Task 1
    exttaskinput1_on=1250*ms
    exttaskinput1_off=1500*ms
    rates1 = '(t>exttaskinput1_on)*(t<exttaskinput1_off)*200*Hz'
    exttaskinput1E=PoissonGroup(1, rates1)
    taskinput1=Synapses(exttaskinput1E, P_E_vACC, 'w : 1', on_pre=eqs_pre_ampa)
    taskinput1.connect()
    taskinput1.w[:] = 100
    
    #Task 2
    exttaskinput2_on=1750*ms
    exttaskinput2_off=2000*ms
    rates2 = '(t>exttaskinput2_on)*(t<exttaskinput2_off)*200*Hz'
    exttaskinput2E=PoissonGroup(1, rates2)
    taskinput2=Synapses(exttaskinput2E, P_E_vACC, 'w : 1', on_pre=eqs_pre_ampa)
    taskinput2.connect()
    taskinput2.w[:] = 100
    
    #Task 3
    exttaskinput3_on=2250*ms
    exttaskinput3_off=2500*ms
    rates3 = '(t>exttaskinput3_on)*(t<exttaskinput3_off)*200*Hz'
    exttaskinput3E=PoissonGroup(1, rates3)
    taskinput3=Synapses(exttaskinput3E, P_E_vACC, 'w : 1', on_pre=eqs_pre_ampa)
    taskinput3.connect()
    taskinput3.w[:] = 100
    
    #Task 4
    exttaskinput4_on=2500*ms
    exttaskinput4_off=2750*ms
    rates4 = '(t>exttaskinput4_on)*(t<exttaskinput4_off)*200*Hz'
    exttaskinput4E=PoissonGroup(1, rates4)
    taskinput4=Synapses(exttaskinput4E, P_E_dlPFC, 'w : 1', on_pre=eqs_pre_ampa)
    taskinput4.connect()
    taskinput4.w[:] = 100
    
    #Task 5
    exttaskinput5_on=3000*ms
    exttaskinput5_off=3250*ms
    rates5 = '(t>exttaskinput5_on)*(t<exttaskinput5_off)*200*Hz'
    exttaskinput5E=PoissonGroup(1, rates5)
    taskinput5=Synapses(exttaskinput5E, P_E_dlPFC, 'w : 1', on_pre=eqs_pre_ampa)
    taskinput5.connect()
    taskinput5.w[:] = 100
    
    #Task 6
    exttaskinput6_on=3500*ms
    exttaskinput6_off=3750*ms
    rates6 = '(t>exttaskinput6_on)*(t<exttaskinput6_off)*200*Hz'
    exttaskinput6E=PoissonGroup(1, rates6)
    taskinput6=Synapses(exttaskinput6E, P_E_dlPFC, 'w : 1', on_pre=eqs_pre_ampa)
    taskinput6.connect()
    taskinput6.w[:] = 100

    #Spike and Population Monitors
    Ex_vACC_SpikeMonitor = SpikeMonitor(P_E_vACC)
    Ex_dlPFC_SpikeMonitor = SpikeMonitor(P_E_dlPFC)
    EVP=PopulationRateMonitor(P_E_vACC)
    EDP=PopulationRateMonitor(P_E_dlPFC)
    
    run(5000*ms, report='stdout')
    
    return (Ex_vACC_SpikeMonitor, Ex_dlPFC_SpikeMonitor, EVP, EDP)

#Initialize spike train and population rate arrays
vACC_spike_train_treat = []
dlPFC_spike_train_treat = []
vACC_population_rate_treat = []
dlPFC_population_rate_treat = []

for i in range(0, 4):
    if i == 0:
        tau_AMPA_vACC = 2.05 * ms
        V_L_T = -70.18 *mV
        g_NMDA_E_vACC = 0.044 * nS 
        Ex_vACC_SpikeMonitor, Ex_dlPFC_SpikeMonitor, EVP, EDP = simulate(tau_AMPA_vACC, V_L_T, g_NMDA_E_vACC)
        vACC_spike_train_treat.append(Ex_vACC_SpikeMonitor.i)
        dlPFC_spike_train_treat.append(Ex_dlPFC_SpikeMonitor.i)
        vACC_population_rate_treat.append(EVP.smooth_rate(window='flat', width=100*ms)/Hz)
        dlPFC_population_rate_treat.append(EDP.smooth_rate(window='flat', width=100*ms)/Hz)
        print(i)
    if i == 1:
        tau_AMPA_vACC = 2.05 * ms
        V_L_T = -70 *mV
        g_NMDA_E_vACC = 0.04 * nS 
        Ex_vACC_SpikeMonitor, Ex_dlPFC_SpikeMonitor, EVP, EDP = simulate(tau_AMPA_vACC, V_L_T, g_NMDA_E_vACC)
        vACC_spike_train_treat.append(Ex_vACC_SpikeMonitor.i)
        dlPFC_spike_train_treat.append(Ex_dlPFC_SpikeMonitor.i)
        vACC_population_rate_treat.append(EVP.smooth_rate(window='flat', width=100*ms)/Hz)
        dlPFC_population_rate_treat.append(EDP.smooth_rate(window='flat', width=100*ms)/Hz)
        print(i)
    if i == 2:
        tau_AMPA_vACC = 2.15 * ms
        V_L_T = -70.6 *mV
        g_NMDA_E_vACC = 0.044 * nS 
        Ex_vACC_SpikeMonitor, Ex_dlPFC_SpikeMonitor, EVP, EDP = simulate(tau_AMPA_vACC, V_L_T, g_NMDA_E_vACC)
        vACC_spike_train_treat.append(Ex_vACC_SpikeMonitor.i)
        dlPFC_spike_train_treat.append(Ex_dlPFC_SpikeMonitor.i)
        vACC_population_rate_treat.append(EVP.smooth_rate(window='flat', width=100*ms)/Hz)
        dlPFC_population_rate_treat.append(EDP.smooth_rate(window='flat', width=100*ms)/Hz)
        print(i)
    if i == 3:
        tau_AMPA_vACC = 2.15 * ms
        V_L_T = -70 *mV
        g_NMDA_E_vACC = 0.04 * nS 
        Ex_vACC_SpikeMonitor, Ex_dlPFC_SpikeMonitor, EVP, EDP = simulate(tau_AMPA_vACC, V_L_T, g_NMDA_E_vACC)
        vACC_spike_train_treat.append(Ex_vACC_SpikeMonitor.i)
        dlPFC_spike_train_treat.append(Ex_dlPFC_SpikeMonitor.i)
        vACC_population_rate_treat.append(EVP.smooth_rate(window='flat', width=100*ms)/Hz)
        dlPFC_population_rate_treat.append(EDP.smooth_rate(window='flat', width=100*ms)/Hz)
        print(i)
                               
#Results Plotting
plt.figure()
plt.title('vACC Average Population Rate')
plt.plot(EVP.t/ms, vACC_population_rate_treat[0], color='black', label='SSRI on Mild MDD')
plt.plot(EVP.t/ms, vACC_population_rate_treat[1], color='blue', label='NMDA receptor antagonist on Mild MDD')
plt.plot(EVP.t/ms, vACC_population_rate_treat[2], color='green', label='SSRI on Severe MDD')
plt.plot(EVP.t/ms, vACC_population_rate_treat[3], color='red', label='NMDA receptor antagonist on Severe MDD')
plt.xlim(0, 5000)
xlabel('Time (ms)')
ylabel('Firing Rate (Hz)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('vACC Population Treat.png', dpi=300)
#plt.show()

plt.figure()
plt.title('dlPFC Average Population Rate')
plt.plot(EVP.t/ms, dlPFC_population_rate_treat[0], color='black', label='SSRI on Mild MDD')
plt.plot(EVP.t/ms, dlPFC_population_rate_treat[1], color='blue', label='NMDA receptor antagonist on Mild MDD')
plt.plot(EVP.t/ms, dlPFC_population_rate_treat[2], color='green', label='SSRI on Severe MDD')
plt.plot(EVP.t/ms, dlPFC_population_rate_treat[3], color='red', label='NMDA receptor antagonist on Severe MDD')
plt.xlim(0, 5000)
xlabel('Time (ms)')
ylabel('Firing Rate (Hz)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('dlPFC Population Treat.png', dpi=300)
#plt.show()

#End of computation time
stop = timeit.default_timer()
print ('Computation Time =', stop - start) 
