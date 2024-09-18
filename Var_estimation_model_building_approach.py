import numpy as np
import pandas as pd

'''
file = open('stock_price_return.txt')
lines = file.readlines()
data = np.loadtxt(lines)

SP_500 = data[:,0]
FTSE = data[:,1]
CAC40 = data[:,2]
Nikkei = data[:,3]

SP_500_return = (np.diff(SP_500)/SP_500[0:500])
FTSE_return = (np.diff(FTSE)/FTSE[0:500])
CAC40_return = (np.diff(CAC40)/CAC40[0:500])
Nikkei_return = (np.diff(Nikkei)/Nikkei[0:500])

print(np.std(SP_500_return))
'''

data1 = pd.read_csv("stock_price_return.txt", sep="\t", header = None)
data1.columns = ["SP_500", "FTSE", "CAC40", "Nikkei"]
#print(data1[0:499])
delta_x = data1.pct_change().dropna() ##diff(axis=0)/(data1[0:499]))

#print(delta_x)

correlation_matrix = delta_x.corr()

#print(correlation_matrix) #"SP_500","FTSE"  .iloc[0,1]

std = delta_x.std(ddof=0)
#print(std)

covar_matrix = correlation_matrix.mul(std, axis=1).mul(std, axis=0)

alpha = [4000, 3000, 1000, 2000]

sigma_p2 = covar_matrix.mul(alpha, axis=0).mul(alpha, axis=1)

sigma_p = np.sqrt(sigma_p2.to_numpy().sum())

print(sigma_p)

print("99% percentile 1-day VaR:", 2.326*sigma_p)
print("99% percentile 10-day VaR:", 2.326*np.sqrt(10)*sigma_p)

#Expected shortfall

mu = 0
Y = 2.326
X = 0.99

ES = mu + sigma_p*((np.exp(-Y**2/2.))/(np.sqrt(2*np.pi)*(1-X)))

print("99% percentile 1-day ES:", ES)
print("99% percentile 10-day ES:", ES*np.sqrt(10))
