import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv("C:/Users/aizawa/Desktop/kuramoto_data4.csv", header = 0)
arr1 = np.array(df1)

m = arr1.size
cost=[]    

alpha = 0.001

a = 5
b = 5
t = 100000

for i in range(t):
    h = a * arr1[:, 0] + b
    cost.append(1 / (2*m) * (np.sum((h - arr1[:, 1])**1)))
    a = a - alpha/m * np.sum((h - arr1[:, 1])*arr1[:, 0])
    b = b - alpha/m * np.sum((h - arr1[:, 1]))
    if cost[i] < 0.015:
        if i < t:
            t = i


plt.subplot(2,1,1)
plt.plot(cost)


print("回帰係数　：　", a)
print("切片　　　　：　", b)
print("収束時間　：　", t)


plt.subplot(2,1,2)
plt.scatter(arr1[:, 0], arr1[:, 1])
plt.plot(arr1[:, 0], h, c="black")


R = np.sqrt(np.sum((h[:] - arr1[:,1].mean())**2) / np.sum((arr1[:,1] - arr1[:,1].mean())**2))
print("相関係数　：　", R)