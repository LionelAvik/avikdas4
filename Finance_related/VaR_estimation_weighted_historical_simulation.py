import numpy as np
import pandas as pd

df = pd.read_csv("stock_price_return.txt", sep="\t", header = None)
df.column = ["SP_500", "FTSE", "CAC40", "Nikkei"]
#print(df.iloc[500])

next_day_stock_price = ((df/df.shift(1))*df.iloc[500]).dropna()
#print(next_day_stock_price) 

alpha = [4000, 3000, 1000, 2000]
current_portfolio_value = sum(alpha)

##neaxt day portfolio value for different scenarios###
portfolio_value = ((next_day_stock_price/df.iloc[500])*alpha).sum(axis=1)
#print(portfolio_value)

##neaxt day gain(Loss) for different scenarios###
gain = portfolio_value - current_portfolio_value
loss = (-1*gain)

scenario = np.arange(1, loss.size+1, 1) #Array of total number of scenarios

lambda1 = 0.995
n = np.size(scenario) #Total number of scenarios

w = lambda1**(n-scenario) * ((1 - lambda1)/(1 - lambda1**n)) #weighted values

C = np.sort(loss)[::-1] ##Descending order (i.e., Max loss to Min loss)
C_Index = np.argsort(loss)[::-1] #Descending order
#print(scenario[C_Index])
w_new = (w[C_Index]) # weight values ordered according to loss order

percentile_Index = C.size - int((99/100)*(C.size)) - 1 # -1 because python array indices start from 0
#print(percentile_Index)

print("99% percentile 1-day VaR (for uniform weighting):", C[percentile_Index])
print("99% percentile 1-day ES (for uniform weighting):", np.mean(C[0:percentile_Index+1]))

cum_w = np.cumsum(w_new)
##print(cum_w)

for i in range(cum_w.size):

    if cum_w[i] >= 0.01:
#        print(i)
        print("99% percentile 1-day VaR (for weighting observation):", C[i])
        ES = (np.sum(w_new[0:i]*C[0:i]) + ((0.01-cum_w[i-1])*C[i]))/0.01
        print("99% percentile 1-day ES (for weighting observation):", ES)
        break





