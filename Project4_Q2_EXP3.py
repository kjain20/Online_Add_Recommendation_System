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
cum_reward = 0

#eta_list = np.zeros(7); #Used to plot for multiple values of eta
#regret_list = np.zeros(7); #Used to plot for multiple values of eta
#for et in range(0, 7): Used to plot for multiple values of eta
loss = np.zeros(k)
p = np.ones(k) / k;
cum_losss = np.zeros(T)
best_column_loss = np.zeros(T)
best_row_loss = np.zeros(T)
cum_loss = 0
#eta_list = [(6/np.sqrt(1)), (3/np.sqrt(1)), (1/np.sqrt(1)), np.sqrt(math.log(k) / ((1)*k)), (2/(1)), (1/(1)), 1/(2*(1))] #Used to plot for multiple values of eta
for i in range(0, T):
#    eta_list = [(6/np.sqrt(i+1)), (3/np.sqrt(i+1)), (1/np.sqrt(i+1)), np.sqrt(math.log(k) / ((i+1)*k)), (2/(i + 1)), (1/(i + 1)), 1/(2*(i + 1))] #Used to plot for multiple values of eta
    eta = 1/np.sqrt(i+1)

    # Update Probability Distribution
    px = np.cumsum(p)
    rand_num = random.random()
    
    # Sample the Ad   
    idx = np.argmax(px >= rand_num)
    reward = data[idx, i]
        
    # Update the Loss
    loss[idx] = loss[idx] + ((1 - reward) / p[idx])
    cum_loss = cum_loss + 1 - data[idx, i]

    # Update Probability Distribution
    p = np.exp(loss * eta * -1)
    p = p / np.sum(p)
    
    # Book keeping        
    cum_losss[i] = cum_loss
    best_column_loss[i] = np.sum(1 - np.max(data[:, 0:i], 0))
    best_row_loss[i] = np.min(np.sum(data[:, 0:i] == 0, 1))
    
#    regret_list[et] = cum_loss - best_row_loss[T-1]    Used to plot for multiple values of eta

# Plot of regret for different values of eta
#plt.plot(eta_list, regret_list, 'ro--')
#plt.xlabel("Eta")
#plt.ylabel("Regret")
#plt.title("Partial Feedback: Eta vs Regret")
#plt.show()

# Plot of probability distribution
#plt.plot(range(k), p)
#plt.xlabel("Ads")
#plt.ylabel("Probability")
#plt.show()

# Plot to visualize Loss, Regret for each Round
plt.plot(range(T), range(T), label = "T")
plt.plot(range(T), cum_losss, label = "Our Loss")
plt.plot(range(T), best_column_loss, label = "Best Column Loss")
plt.plot(range(T), best_row_loss, label = "Best Row Loss")
plt.plot(range(T), (cum_losss - best_column_loss), 'r--', label = "Optimistic Regret")
plt.plot(range(T), (cum_losss - best_row_loss), 'y--', label = "Regret")
plt.legend()
plt.show()

# Plot to visualize regret independently
plt.plot(range(T), (cum_losss - best_row_loss), 'y--', label = "Regret")
plt.legend()
plt.show()

print("Regret: {}", cum_losss[T-1] - best_row_loss[T-1])
