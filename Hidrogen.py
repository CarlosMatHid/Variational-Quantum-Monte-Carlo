"""

                                HYDROGEN
                                
                     Joana Fraxanet and Carlos Mateos

"""

import numpy as np
import matplotlib.pyplot as plt

#Define trial function
def wf(r,alpha):
    '''Computes the trial wavefunction'''
    r_norm = np.linalg.norm(r)
    wf = np.exp(-alpha*r_norm)
    return wf

#define prob density
def prob_density(r,alpha):
    '''Computes the probability density (not normalized) of the trial 
    wavefunction'''
    return wf(r,alpha)**2

#define E local
def E_local(r,alpha):
    '''Computes the local energy, in terms of r1, r2 and alpha, corresponding
    to the trial wavefunction'''
    r_norm = np.linalg.norm(r)
    energy = -1/r_norm-alpha*(alpha-2/r_norm)/2
    return energy 
    
    
def metropolis(N, alpha):
    '''Metropolis algorithm that takes N steps. We start with two random 
    variable within the typical length of the problem and then we create a 
    Markov chain taking into account the probability density. At each step we
    compute the parameters we are interested in.'''
        
    L = 2
    r = (np.random.rand(3)*2*L-L)
    E = 0
    E2 = 0 #it is going to be E**2 (useful to compute variance of energy)
    Eln_average = 0
    ln_average = 0
    rejection_ratio = 0
    #Algorithm
    for i in range(N):
        r_trial = r + 0.2*(np.random.rand(3)*2*L-L)
        if prob_density(r_trial,alpha) >= prob_density(r,alpha):
            r = r_trial
        else:
            dummy = np.random.rand()
            if dummy < prob_density(r_trial,alpha)/prob_density(r,alpha):
                r = r_trial
            else:
                rejection_ratio += 1/N
                
        E += E_local(r,alpha)/N
        E2 += E_local(r,alpha)**2/N
        Eln_average += E_local(r,alpha)*-np.linalg.norm(r)/N
        ln_average += -np.linalg.norm(r)/N
    
    return E, E2, Eln_average, ln_average, rejection_ratio

    
'''Initial parameters'''
alpha = 0.5
#alpha_iterations = 20
alpha_iterations = 50
N_metropolis = 500
random_walkers = 200
gamma = 0.5 

energy_plot = np.array([])
alpha_plot = np.array([])
variance_plot = np.array([])

for i in range(alpha_iterations):
    '''This loop iterates over alpha to find the value which minimizes the Energy.
    Apart from that, it calculates the rejection ratio and saves data needed for plotting'''
    
    E = 0
    E2 = 0
    dE_dalpha = 0
    Eln = 0
    ln = 0
    rejection_ratio = 0
    for j in range(random_walkers): #We use more than one random_walkers in case one gets stuck at some X
        E_met, E2_met, Eln_met, ln_met, rejections_met = metropolis(N_metropolis, alpha)
        E += E_met/random_walkers
        E2 += E2_met/random_walkers
        Eln += Eln_met/random_walkers
        ln += ln_met/random_walkers
        rejection_ratio += rejections_met/random_walkers
        
    print('Alpha: ', alpha, '<E>: ', E, 'VarE: ', E2-E**2, 'ratio = ', rejection_ratio)

    # Define next alpha
    dE_dalpha = 2*(Eln-E*ln)
    alpha = alpha - gamma*dE_dalpha
#    alpha = alpha + 0.05

    # plots:    
    energy_plot = np.append(energy_plot, E)
    alpha_plot = np.append(alpha_plot, alpha)
    variance_plot = np.append(variance_plot, E2-E**2)
    
fig1 = plt.figure()

ax1 = fig1.add_subplot(311)
plt.title('HYDROGEN ATOM: evolution of the parameters')
plt.grid()
ax1.plot(alpha_plot, 'g')
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Alpha')

ax2 = fig1.add_subplot(312)
plt.grid()
ax2.plot(energy_plot)
ax2.set_xlabel('Timestep')
ax2.set_ylabel('E exp value')
ax2.errorbar(range(len(energy_plot)), energy_plot, yerr=np.sqrt(variance_plot), c='b')

ax3 = fig1.add_subplot(313)
plt.grid()
ax3.plot(variance_plot, 'r')
ax3.set_xlabel('Timestep')
ax3.set_ylabel('Var E')

fig2 = plt.figure()
ax4 = fig2.add_subplot(111)
plt.title('Hydrogen atom: Energy vs. alpha')
plt.grid()
ax4.plot(alpha_plot, energy_plot, 'ro')
ax4.set_xlabel('Alpha')
ax4.set_ylabel('Energy')

