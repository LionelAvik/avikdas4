import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import sys
import matplotlib.pyplot as plt
import random as rd

'''

Chang-Cooper algorithm to solve diffusion-loss equation (Fokker-Planck equation) for arbitrary injected particle spectrum. This code is written just to understand simply that how the algorithm works. 
Reference: "A practical difference scheme for Fokker-Planck equations" (url: https://www.sciencedirect.com/science/article/abs/pii/002199917090001X?via%3Dihub)

'''

gamma_min = 50.0 #1.0
gamma_max = 1000.0 #1.0e7
n_steps = 300
gamma_step = np.exp((1.0 / (n_steps)) * np.log(gamma_max/gamma_min))

gamma_grid = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_steps, endpoint=False)
half_gamma_grid = (gamma_grid[0:n_steps-1]+gamma_grid[1:n_steps])/2.0
#print(gamma_grid, half_gamma_grid)

first_delta_gamma_grid = gamma_grid[0]*(1 - (1/gamma_step)) 
last_delta_gamma_grid = gamma_grid[-1]*(gamma_step - 1)

delta_gamma_grid = np.append(first_delta_gamma_grid, np.diff(gamma_grid))
delta_gamma_grid = np.append(delta_gamma_grid, last_delta_gamma_grid)

#print(delta_gamma_grid)

delta_gamma_grid_bar = (delta_gamma_grid[0:n_steps]+delta_gamma_grid[1:n_steps+1])/2.0

#print(delta_gamma_grid_bar)

tau0 = 10.0
def escape_time(x):
	return (tau0)*x**0
tau_e = escape_time(gamma_grid)

Q0 = 1.0
x_min = 50.0
x_max = 1000.0
p = -3.0

'''
x0 = gamma_grid[150]
def injection_term(x): #Mono-energetic electron injection term
	if x == x0:
		return Q0 
	else:
		return 0
'''

def injection_term(x): #Power-law injection

	if (x_min <= x) & (x <= x_max):

		return Q0 *((x_max**(p+1.0) - x_min**(p+1.0))/(p+1.0))* np.power(x, p)		
	else:
	
		return 0.0

	
A1 = 1.0
A2 = 0.0  #0.0000001
tau_a =  100.0 #0.0001

B = np.zeros(half_gamma_grid.size)
C = np.zeros(half_gamma_grid.size)

def heating_term(x): # Bj's
	return A1*x**2 - (x/tau_a) #(A1*x**2)  
def diffusion_term(x): # Cj's
	return A2*(1/(2*tau_a))*(x**2)
 
B = heating_term(half_gamma_grid)  ## term corresponding to first derivative (n_steps-1)
C = diffusion_term(half_gamma_grid) ## term corresponding to second derivative (n_steps-1)

time_min = Decimal('0.0')
time_max = Decimal('0.01')
delta_time_step = Decimal('0.001')
time_grid = np.arange(float(time_min), float(time_max+delta_time_step), float(delta_time_step))
print(time_grid)

##Compute delta_j
delta_j = np.zeros(n_steps-1)
for j in range(n_steps-1):
	if C[j] !=0: #dispersion term at half grid points
		w = (B[j]/C[j])*delta_gamma_grid[1:-1][j]
		if w==0:
			delta_j[j] = 0.5
		else:
			delta_j[j] = (1.0/w) - 1.0/(np.exp(w)-1.0)
	else:
		delta_j[j] = 0
#print(delta_j)

#equation bj*uj = aj*u_j+1 + cj*u_j-1 + dj

a = np.zeros(n_steps)
b = np.zeros(n_steps)
c = np.zeros(n_steps)

for j in range(n_steps-2, 0, -1):
	one_over_delta_gamma_grid_forward = 1.0/(delta_gamma_grid[j+1])
	one_over_delta_gamma_grid_backward = 1.0/(delta_gamma_grid[j])

	one_over_delta_gamma_grid_bar = 1.0/(delta_gamma_grid_bar[j])	
	
	B_forward = B[j]
	B_backward = B[j-1]
	
	C_forward = C[j]
	C_backward = C[j-1]
	
	a[j] = float(delta_time_step)*one_over_delta_gamma_grid_bar*((1-delta_j[j])*B_forward + (one_over_delta_gamma_grid_forward*C_forward)) #n+1_j+1 term
	
	b[j] = 1.0 + float(delta_time_step)*one_over_delta_gamma_grid_bar*((one_over_delta_gamma_grid_forward*C_forward) + (one_over_delta_gamma_grid_backward*C_backward) + ((1 - delta_j[j-1])*B_backward) - (delta_j[j]*B_forward)) #n+1_j term
	
	c[j] = float(delta_time_step)*one_over_delta_gamma_grid_bar*(one_over_delta_gamma_grid_backward*C_backward - delta_j[j-1]*B_backward) #n+1_j-1 term
	
	
#right boundary condition

a[-1] = 0 #n+1_j+1 term

b[-1] = 1 + (float(delta_time_step)/delta_gamma_grid_bar[-1])*((C[-1]/delta_gamma_grid[-1]) + (1-delta_j[-1])*B[-1]) #n+1_j term

c[-1] = (float(delta_time_step)/delta_gamma_grid_bar[-1])*(C[-1]/delta_gamma_grid[-1] - delta_j[-1]*B[-1]) #n+1_j-1 term

#left boundary condition

a[0] = (float(delta_time_step)/delta_gamma_grid_bar[0])*((1 - delta_j[0])*B[0] + C[0]/delta_gamma_grid[0]) #n+1_j+1 term

b[0] = 1 + (float(delta_time_step)/delta_gamma_grid_bar[0])*((C[0]/delta_gamma_grid[0]) - delta_j[0]*B[0]) #n+1_j term

c[0] = 0 #n+1_j-1 term

##b = b + (float(delta_time_step)/tau_e) # Adding escape term to n+1_j term
		
	
#Tridiagonal Solver	
#equation bj*uj = aj*u_j+1 + cj*u_j-1 + dj
		
u = np.zeros(n_steps)  #Intial Distribution at t=0 (1st row of u_final)

for i in range(30):  # Modification of initial distribution
	u[i + 1] = 0.0  

P = np.zeros(n_steps)
Q = np.zeros(n_steps)

u_final = np.zeros((time_grid.size, n_steps))
u_final[0,:] = u

#print(u_final)
for t in range(1, time_grid.size):
#print(P, Q)
	#u_final[t,:] = u_final[t,:] + injection_term(gamma_grid)*float(delta_time_step)
	for j in range(1, n_steps-1):
		P[0] = a[0]/b[0]
		Q[0] = (u_final[t-1,0] + (injection_term(gamma_grid[0])*float(delta_time_step)))/b[0]
		
		e = b[j] - c[j]*P[j-1]
		P[j] = a[j]/e
		Q[j] = ((u_final[t-1,j]+ (injection_term(gamma_grid[j])*float(delta_time_step)))+c[j]*Q[j-1])/e	
#print(P, Q)	


	u_final[t,n_steps-1] = ((u_final[t-1,n_steps-1] + (injection_term(gamma_grid[n_steps-1])*float(delta_time_step))) + c[n_steps-1]*Q[n_steps-2])/(b[n_steps-1] - c[n_steps-1]*P[n_steps-2])

	for j in range(n_steps-2, -1, -1):
		u_final[t,j] = P[j]*u_final[t,j+1] + Q[j]		
	#print(u)	
	
print(u_final)

for t in range(time_grid.size):
	plt.plot(gamma_grid, u_final[t,:])
#plt.plot(gamma_grid, u_final[-1,:])
	
#plt.legend([f't= {np.round(value,3)}' for value in time_grid], fontsize=15)

#plt.xlim(0.0, 2.0)
plt.xscale('log')
plt.yscale('log')
	
plt.show()	
