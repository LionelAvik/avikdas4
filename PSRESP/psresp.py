#import simulate_lc
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
from scipy.interpolate import interp1d
from multiprocessing import Pool
import sys

'''
PSRESP method to model (Power Spectral Density) PSD with power-law type function (i.e., nu^(-beta) + P_noise; P_noise = Poisson noise)
'''


file_no = 1000 #input("Enter the number of simulated light curves in your directory:")
t_bin = 1.0 #Time bin of your light curve
beta = np.arange(0.2, 2.5, 0.1)
len_beta = int(beta.size)
#print(len_beta)

file = open('sample_lightcurve.txt')
lines = file.readlines()
data = np.loadtxt(lines)

T = data[:,0]
flux_obs = data[:,1]
fluxerr= data[:,2]

mean_lc = np.mean(flux_obs)
sigma = np.std(flux_obs)

#print(mean_lc, sigma)
######### (poission_noise)

t_step = T[1] - T[0]  

N_obs = T.size

err2 = fluxerr**2

mean_err2 = np.mean(err2)
mu = np.mean(flux_obs)

P_noise = ((2*t_step)/(N_obs*mu)**2)*mean_err2

#print(P_noise)
##########
start_time = np.min(T)
end_time  = np.max(T)


flux1 = interp1d(T, flux_obs, fill_value="extrapolate")

reg_Time = np.arange(start_time, end_time+t_bin, t_bin)
reg_flux = flux1(reg_Time)

##plt.plot(T, flux_obs,'bo')
##plt.plot(reg_Time, reg_flux, 'm^')

##plt.show()

#############

nn = reg_Time.size
delt = t_bin

#Fourier frequencies
fi = np.arange(1, nn/2+1, dtype='float64')/nn/delt 

a = 1. #params[0]
nu_0 = 1. #params[1]
#beta = params[2]
noise = P_noise #params[3]

#########

f_obs, pxx_den_obs = signal.periodogram(reg_flux,fs=1,window='hann',scaling='spectrum') #,detrend='linear')
##plt.loglog(f_obs, pxx_den_obs,'ro')

freq_number = f_obs.size #- 1

############### (Timmer & Koineg)

def lc_sim(nn, delt, mean_lc, params):
	
	
	#Fourier frequencies
	fi = np.arange(1, nn/2+1, dtype='float64')/nn/delt 
	#print fi
	
	
	a = params[0]
	nu_0 = params[1]
	beta = params[2]
	noise = params[3]
		
	s = a*(fi/nu_0)**(-beta) + noise
#		print(s)
		
	
	#Generate two sets of normally distributed random numbers 	
	aa = np.random.randn(len(fi))
	bb = np.random.randn(len(fi))

	
	#Fourier transform of light curve  = SQRT(S/2) * (A + B*i)
	flc = np.sqrt(.5*s)*(aa + bb*1.j)
	
	if np.mod(nn, 2) == 0:
		flc[-1] = np.sqrt(.5*s[-1])*1
	del aa, bb, s
	

	#Put the mean of the light curve at frequency = 0
	flc = np.hstack([mean_lc, flc])
#	print(flc)
	
	#Take the inverse fourier transform to generate synthetic light curve
	lc = np.fft.irfft(flc, n=nn)
	
	return lc


##########

success_frac = np.zeros(beta.size)
#print(success_frac.size)

for k in range(len_beta):


#	TK = np.zeros((reg_Time.size, reg_flux.size))
	
	t = np.zeros((file_no,freq_number,2)) #20
	t_sq = np.zeros((file_no,freq_number,2))

	for i in range(file_no):
		
	
		params = [1.0, 1.0, beta[k], P_noise]

		sim_lc = lc_sim(nn, delt, mean_lc, params)

#		TK = np.zeros((reg_Time.size, reg_flux.size))
	

#		plt.plot(reg_Time, sim_lc)

#		plt.show()
		avg_sim = np.mean(sim_lc)
		std_sim = np.std(sim_lc)

#sim_lc = (sim_lc - avg) * sigma / std + mean_lc

		tempvar = np.sqrt(np.var(sim_lc))

		lc_variance = np.var(flux_obs) #- np.var(fluxerr))

		sim_lc = (sim_lc - avg_sim)*((np.sqrt(lc_variance))/tempvar) + mean_lc
#		print(sim_lc)

		f, pxx_den = signal.periodogram(sim_lc,fs=1,window='hann',scaling='spectrum')#,detrend='linear')


		t[i,:,0] = f[:]
		t[i,:,1] = pxx_den[:] #pxx_den_new[:]
#	t_sq[i,:,1] = data1[:,1]*data1[:,1]

	mean_P = np.mean(t,axis=0)
	std_P = np.std(t,axis=0)

	chi_sim = np.zeros(file_no)
	chi_obs = np.zeros(file_no)
	for i in range(file_no):
		C = (((t[i,:,1] - mean_P[:,1])**2)/(std_P[:,1])**2)
		C_sum = np.sum(C)	
		chi_sim[i] = C_sum
		K = (((pxx_den_obs - mean_P[:,1])**2)/(std_P[:,1])**2)
		K_sum = np.sum(K)
		chi_obs[i] = K_sum


	A = chi_sim/chi_obs
#	print(A)
#	print(chi_obs.size)
	r = 1
	count = 0

	for j in A:
		if j >= r:
			count = count + 1
#	print(count)
	success_frac[k] = (float(count) / chi_sim.size)*100
	print(round(beta[k],1), round(success_frac[k],2))
	
	del t, t_sq
#	return success_frac

np.savetxt("beta_value.txt", np.transpose([beta, success_frac]), fmt="%1.2f\t%1.2f")


######fitting success fraction with Gaussian function#####
#sigma = 13.6
#x0 = 0.8
#A = 100

x = beta
y = success_frac

def func(x,A,sigma, x0): 
    return A*np.exp(-np.power((x - x0)/sigma, 2.)/2)

popt,pcov = curve_fit(func,x,y,p0 =[100,0.4,0.8],maxfev=1000) # Initial value of the parameters, change this if you are using different light curves 


perr=np.sqrt(np.diag(pcov))

##print (popt)
##print (perr)

print(r"Best fitted beta value (index of the powerlaw): %s" %np.round(popt[2],3))
print(r"uncertainty on beta value (index of the powerlaw): %s" %np.round(popt[1],3))

fig = plt.figure(figsize=(8,5))

plt.plot(x,y,'bo',lw=4,markersize=15)

x_fit = np.linspace(0.1,2.5,100)
yfit = func(x,*popt)
y_fit=func(x_fit,*popt)

plt.plot(x_fit,y_fit,'k',lw=4.5)


bound_upper1 = func(x_fit, *(popt + 1.306*perr))
bound_lower1 = func(x_fit, *(popt - 1.306*perr))

bound_upper2 = func(x_fit, *(popt + 2.434*perr))
bound_lower2 = func(x_fit, *(popt - 2.434*perr))


plt.fill_between(x_fit, bound_lower1, bound_upper1, color = 'red', alpha = 1.0, label = '90% CI')

plt.fill_between(x_fit, bound_lower2, bound_upper2, color = 'grey', alpha = 0.7, label = '99% CI')

plt.xlabel(r'Value of $\beta$',fontsize = 25)
plt.ylabel(r'Success Fraction (%)',fontsize = 25)

plt.legend(loc='upper right',fontsize=20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

#plt.show()
