"""

                                HARMONIC OSCILLATOR
                                
                        Joana Fraxanet and Carlos Mateos

"""

import numpy as np
import matplotlib.pyplot as plt


#Define trial function
def wf(x,alpha):
    '''Computes the trial wavefunction'''
    return np.exp(-alpha*x**2)

#define prob density
def prob_density(x,alpha):
    '''Computes the probability density (normalized) of the trial wavefunction'''
    return wf(x,alpha)**2/np.sqrt(np.pi/alpha)

#define E local
def E_local(x,alpha):
    '''Computes the local energy, in terms of x and alpha, corresponding to the trial wavefunction'''
    return alpha+x**2*(1/2-2*alpha**2)
    
def metropolis(N, alpha):
    '''Metropolis algorithm that takes N steps. We start with a random variable within the typical length of the problem and then we create a Markov chain taking into account the probability density. At each step we compute the parameters we are interested in.'''
    
    L = 3/np.sqrt(2*alpha) #3 times the typical length (wavefunction is a Gaussian). Three sigmas
    x = np.random.rand()*2*L-L #random number from -L to L
    E = 0
    E2 = 0
    Eln_average = 0
    ln_average = 0
    rejection_ratio = 0
    #Algorithm
    for i in range(N):
        x_trial = x + 0.4*(np.random.rand()*2*L-L)
        #x_trial = np.random.rand()*2*L-L 
        if prob_density(x_trial,alpha) >= prob_density(x,alpha):
            x = x_trial
        else:
            dummy = np.random.rand()
            if dummy < prob_density(x_trial,alpha)/prob_density(x,alpha):
                x = x_trial
            else:
                rejection_ratio += 1/N
        E += E_local(x,alpha)/N
        E2 += E_local(x,alpha)**2/N
        Eln_average += (E_local(x, alpha)*-x**2)/N
        ln_average += -x**2/N
    
    return E, E2, Eln_average, ln_average, rejection_ratio
    
'''Initial parameters'''
alpha = 0.1
#alpha = 1.2
#alpha_iterations = 20
alpha_iterations = 20
N_metropolis = 500
random_walkers = 200
gamma = 0.9

energy_plot = np.array([])
alpha_plot = np.array([])
variance_plot = np.array([])
E_analytical_plot = np.array([])
var_analytical_plot = np.array([])

'''Iterations for alpha'''
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

    '''Analytical expressions for E and variance to compare'''
    E_analytical = alpha/2 + 1/8/alpha
    var_analytical = (1-4*alpha**2)**2/32/alpha**2

    '''Define next alpha'''
    dE_dalpha = 2*(Eln-E*ln)
    print('Alpha: %0.4f' % alpha, '<E>: %0.4f' % E, 'E_analytical: % 0.4f' % E_analytical, 'VarE: %0.4f' % (E2-E**2), 'Var_analytical: %0.4f' % var_analytical, 'rejection ratio: ',rejection_ratio)
#    alpha = alpha - gamma*dE_dalpha
    alpha = alpha + 0.05
    
    '''Plot'''
    energy_plot = np.append(energy_plot, E)
    alpha_plot = np.append(alpha_plot, alpha)
    variance_plot = np.append(variance_plot, E2-E**2)
    E_analytical_plot = np.append(E_analytical_plot, E_analytical)
    var_analytical_plot = np.append(var_analytical_plot, var_analytical)


fig1 = plt.figure()


ax1 = fig1.add_subplot(311)
plt.title('Harmonic Oscillator: Minimization process')
plt.grid()
ax1.plot(alpha_plot, 'g')
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Alpha')

ax2 = fig1.add_subplot(312)
plt.grid()
ax2.plot(energy_plot, 'b')
ax2.set_xlabel('Timestep')
ax2.set_ylabel('E exp value')
ax2.errorbar(range(len(energy_plot)), energy_plot, yerr=np.sqrt(variance_plot), c= 'b')

ax3 = fig1.add_subplot(313)
plt.grid()
ax3.plot(E_analytical_plot, 'r')
ax3.errorbar(range(len(energy_plot)), E_analytical_plot, yerr=np.sqrt(var_analytical_plot), c = 'r')
ax3.set_xlabel('Timestep')
ax3.set_ylabel('E exp value analytical')

fig2 = plt.figure()

ax1 = fig2.add_subplot(311)
plt.title('Harmonic Oscillator: Minimization process')
plt.grid()
ax1.plot(alpha_plot, 'g')
ax1.set_ylabel('Alpha')
ax1.set_xlabel('Timestep')

ax2 = fig2.add_subplot(312)
plt.grid()
ax2.plot(variance_plot, 'b')
ax2.set_ylabel('Var E')
ax2.set_xlabel('Timestep')

ax3 = fig2.add_subplot(313)
plt.grid()
ax3.plot(var_analytical_plot, 'r')
ax3.set_ylabel('Var E analytical')
ax3.set_xlabel('Timestep')

fig3 = plt.figure()
ax1 = fig3.add_subplot(111)
plt.title('Harmonic Oscillator: Energy vs. alpha')
plt.grid()
ax1.plot(alpha_plot, energy_plot, 'ro', label = 'Numerical')
ax1.plot(alpha_plot, E_analytical_plot, 'bo', label = 'Analytical')
plt.legend()
ax1.set_xlabel('Alpha')
ax1.set_ylabel('Energy')
ax1.set_ylim(0.4, 1.5)
