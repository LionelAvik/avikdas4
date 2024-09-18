import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("stock_price_return.txt", sep = "\t", header = None)
df.columns = ["SP_500", "FTSE", "CAC40", "Nikkei"]
#print(df)

delta_x = df.pct_change().dropna()
#print(delta_x)

mean = np.array([0,0,0,0]) #Mean value of each market variable (assumed to be zero)
std = delta_x.std(ddof=0) #standard deviation each market variable
#print(std)

correlation_matrix = delta_x.corr()
#print(correlation_matrix)

covariance_matrix = correlation_matrix.mul(std, axis=0).mul(std, axis=1).to_numpy()
#print(covariance_matrix)

current_portfolio_value = 10000 #Current value of the portfolio

alpha = [4000, 3000, 1000, 2000] #amount invested on a given day for each market variable

#np.random.seed(42)
sim_size = 10000
delta_x_sim = np.random.multivariate_normal(mean, covariance_matrix, size=sim_size)

market_variable_value = alpha + alpha*delta_x_sim

#print(market_variable_value)

portfolio_value_sim = np.sum(market_variable_value, axis=1)

#print(portfolio_value_sim)

delta_P = (portfolio_value_sim - current_portfolio_value)*(-1)

####plotting the distribution of Delta_P####
#plotting histogram of delta_P (i.e., Gain distrtibution)#
plt.hist(delta_P, bins=30, color = "blue") 
plt.xlabel("Gain (Loss)", fontsize=20)
plt.ylabel("Count", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

###99% percentile VaR calculation (For loss distribution, we need it compute 99% percentile)

delta_P_sorted = np.sort(delta_P)[::-1] #sorted from worst loss (+ve) to least loss (-ve)
#print(delta_P_sorted)
desired_Index = int(sim_size*(1 - 0.99)) - 1
print("99% percentile 1-day VaR:", delta_P_sorted[desired_Index])
print("99% percentile 10-day VaR:", delta_P_sorted[desired_Index]*np.sqrt(10))

print("99% percentile 1-day ES:", np.mean(delta_P_sorted[0:desired_Index+1]))
print("99% percentile 1-day ES:", np.mean(delta_P_sorted[0:desired_Index+1])*np.sqrt(10))

plt.show()
