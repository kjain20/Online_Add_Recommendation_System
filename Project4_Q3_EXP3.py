import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse import find
import csv
import random
from random import randint
import matplotlib.pyplot as plt 
import math

start_time = time.time()

inputData = pd.read_csv('yahoo_ad_clicks.csv', header = None)
k = inputData.shape[0]
T = inputData.shape[1]
data = np.array(inputData)

weight = np.ones(k)
cum_reward = 0
cum_loss = 0

#eta_list = np.zeros(5); #Used to plot for multiple values of eta
#regret_list = np.zeros(5); #Used to plot for multiple values of eta
#for et in range(0, 5): #Used to plot for multiple values of eta
loss = np.zeros(k)
p = np.ones(k) / k;
weight = np.ones(k)
cum_losss = np.zeros(T)
best_column_loss = np.zeros(T)
best_row_loss = np.zeros(T)
cum_loss = 0
#   eta_list = [(1/np.sqrt(2)), np.sqrt(math.log(k) / ((1)*k)), (2/(1)), (1/(1)), 1/(2*(1))]#Used to plot for multiple values of eta

for i in range(0, T):
#   eta_list = [(1/np.sqrt(i+3)), np.sqrt(math.log(k) / ((i+2)*k)), (2/(i + 3)), (1/(i + 3)), 1/(2*(i + 3))]#Used to plot for multiple values of eta
#   eta = eta_list[et]#Used to plot for multiple values of eta
    eta = 1/np.sqrt(i+2)
    
    # Update Probability Distribution
    p =  weight / np.sum(weight)
    px = np.cumsum(p)
    
    # Sample the Ad    
    rand_num = random.random()
    idx = np.argmax(px >= rand_num)
    
    # Update Reward   
    cum_reward = cum_reward + data[idx, i]
    cum_loss = cum_loss + 1 - data[idx, i]
    
    # Update All Losses
    for j in range(0, k):
        weight[j] = weight[j] * (1 - (eta * ( 1 - data[j, i])))
    
    cum_losss[i] = cum_loss
    best_row_loss[i] = np.min(np.sum(data[:, 0:i] == 0, 1))
        
#    regret_list[et] = cum_loss - best_row_loss[T-1] #Used to plot for multiple values of eta
        
# Regret Numbers
print("Regret: {}", cum_losss[T-1] - best_row_loss[T-1])

# Plot of regret for different values of eta 
#plt.plot(eta_list, regret_list, 'ro--')
#plt.xlabel("Eta")
#plt.ylabel("Regret")
#plt.title("Full Feedback: Eta vs Regret")
#plt.show()

# Probability Distribution after t Rounds
plt.plot(range(k), p)
plt.xlabel("Ads")
plt.ylabel("Probability")
plt.show()

# Loss, Regrets, T to visualize sublinearity
plt.plot(range(T), range(T), label = "T")
plt.plot(range(T), cum_losss, label = "Our Loss")
plt.plot(range(T), best_column_loss, label = "Best Column Loss")
plt.plot(range(T), best_row_loss, label = "Best Row Loss")
plt.plot(range(T), (cum_losss - best_column_loss), 'r--', label = "Optimistic Regret")
plt.plot(range(T), (cum_losss - best_row_loss), 'y--', label = "Regret")
plt.legend()
plt.show()

# Regret
plt.plot(range(T), (cum_losss - best_row_loss), 'y--', label = "Regret")
plt.legend()
plt.show()  