"""

                                HELIUM

                    Joana Fraxanet and Carlos Mateos
"""

import numpy as np
import matplotlib.pyplot as plt


#Define trial function
def wf(r1,r2,alpha):
    '''Computes the trial wavefunction'''
    norm_r1 = np.linalg.norm(r1)
    norm_r2 = np.linalg.norm(r2)
    r12 = np.linalg.norm(r1-r2)
    wf = np.exp(-2*norm_r1)*np.exp(-2*norm_r2)*np.exp(r12/(2*(1+alpha*r12)))
    return wf

#define prob density
def prob_density(r1,r2,alpha):
    '''Computes the probability density (not normalized) of the trial wavefunction'''
    return wf(r1,r2,alpha)**2

#define E local
def E_local(r1,r2,alpha):
    '''Computes the local energy, in terms of r1, r2 and alpha, corresponding to the trial wavefunction'''
    norm_r1 = np.linalg.norm(r1)
    norm_r2 = np.linalg.norm(r2)
    r12 = np.linalg.norm(r1-r2)        
    dot_product = np.dot(r1/norm_r1-r2/norm_r2,r1-r2)
    energy = -4+dot_product/(r12*(1+alpha*r12)**2)-1/(r12*(1+alpha*r12)**3)-1/(4*(1+alpha*r12)**4)+1/r12 
    return energy
   
def metropolis(N, alpha):
    '''Metropolis algorithm that takes N steps. We start with two random variable within the
    typical length of the problem and then we create a Markov chain taking into account the 
    probability density. At each step we compute the parameters we are interested in.'''
        
    L = 1
    r1 = np.random.rand(3)*2*L-L
    r2 = np.random.rand(3)*2*L-L #random number from -L to L
    E = 0
    E2 = 0
    Eln_average = 0
    ln_average = 0
    rejection_ratio = 0
    step = 0
    max_steps = 500
    
    #Algorithm
    for i in range(N):
        chose = np.random.rand()
        step = step + 1
        if chose < 0.5:
            r1_trial = r1 + 0.5*(np.random.rand(3)*2*L-L)
            r2_trial = r2
        else:
            r2_trial = r2 + 0.5*(np.random.rand(3)*2*L-L)
            r1_trial = r1
        if prob_density(r1_trial,r2_trial,alpha) >= prob_density(r1,r2,alpha):
            r1 = r1_trial
            r2 = r2_trial
        else:
            dummy = np.random.rand()
            if dummy < prob_density(r1_trial,r2_trial,alpha)/prob_density(r1,r2,alpha):
                r1 = r1_trial
                r2 = r2_trial
            else:
                rejection_ratio += 1./N
                
        if step > max_steps:
            E += E_local(r1,r2,alpha)/(N-max_steps)
            E2 += E_local(r1,r2,alpha)**2/(N-max_steps)
            r12 = np.linalg.norm(r1-r2)
            Eln_average += (E_local(r1,r2,alpha)*-r12**2/(2*(1+alpha*r12)**2))/(N-max_steps)
            ln_average += -r12**2/(2*(1+alpha*r12)**2)/(N-max_steps)
    
    return E, E2, Eln_average, ln_average, rejection_ratio

'''Initial parameters'''
alpha = 0
#alpha_iterations = 30
alpha_iterations = 6
N_metropolis = 5000
random_walkers = 200
gamma = 0.5

energy_plot = np.array([])
alpha_plot = np.array([])
variance_plot = np.array([])

for i in range(alpha_iterations):
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

    '''Define next alpha'''
    dE_dalpha = 2*(Eln-E*ln)
    print('Alpha: ', alpha, '<E>: ', E, 'VarE: ', E2-E**2, 'ratio = ', rejection_ratio)
    alpha = alpha + 0.05
    #alpha = alpha - gamma*dE_dalpha

    '''Plot'''    
    energy_plot = np.append(energy_plot, E)
    alpha_plot = np.append(alpha_plot, alpha)
    variance_plot = np.append(variance_plot, E2-E**2)


fig1 = plt.figure()

ax1 = fig1.add_subplot(311)
plt.title('Helium atom: evolution of the parameters')
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
plt.title('Helium atom: Energy vs. alpha')
plt.grid()
ax4.plot(alpha_plot, energy_plot, 'ro')
ax4.set_xlabel('Alpha')
ax4.set_ylabel('Energy')