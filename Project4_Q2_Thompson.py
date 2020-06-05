import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

inputData = pd.read_csv('yahoo_ad_clicks.csv', header = None)
k = inputData.shape[0]
T = inputData.shape[1]
data = np.array(inputData)

prevCumLoss = 0
cumReward = 0

regret = np.zeros(T)
cum_losss = np.zeros(T)
best_row_loss = np.zeros(T)
prob = []

s = np.zeros(k)
f = np.zeros(k)
cum_loss = 0

for i in range(0,T):
    dist = np.random.beta(s+1,f+1)
    index = np.argmax(dist)
    reward = data[index, i]
    if reward == 1:
        s[index] = s[index] + 1
    else:
        f[index] = f[index] + 1
    
    cum_loss = cum_loss + 1 - reward
    cum_losss[i] = cum_loss
    best_row_loss[i] = np.min(np.sum(data[:, 0:i] == 0, 1))
    regret[i] = cum_losss[i] - best_row_loss[i]

plt.plot(regret, 'r--', label = "Regret")
plt.plot(cum_losss, label = "Our Loss")
plt.plot(best_row_loss, label = "Optimal Loss")
plt.xlabel("Rounds")
plt.title("Partial Feedback: Time vs Loss and Regret")
plt.legend()
plt.show()

plt.plot(regret, label = "Regret")
plt.xlabel("Rounds")
plt.ylabel("Regret")
plt.title("Partial Feedback: Time vs Regret")
plt.legend()
plt.show()
print("Regret: {}", regret[T-1])
