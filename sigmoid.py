import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# データセット
# =============================================================================
df1 = pd.read_csv("C:/Users/aizawa/Desktop/programing/2018 1101_zirou/features.csv", header = 0)
arr2 = np.array(df1)
#arr3 = np.arange(-3.5, 4.5)

arr1 = arr2
arr2[:, 0] = (arr1[:, 0] - np.mean(arr1[:, 0])) / np.std(arr1[:, 0])
arr2[:, 1] = (arr1[:, 1] - np.mean(arr1[:, 1])) / np.std(arr1[:, 1])



m = arr2.size
A0 = 1
A1 = 1
A2 = 1
cost = []
t = 100000
alpha = 1


for i in range(t):
    A = A0 + A1 * arr2[:, 0] + A2 * arr2[:, 1]
    h = 1 / (1 + np.exp(A))
    cost.append( 1 / m * np.sum(arr2[:, 2] * h + (1 - arr2[:, 2]) * (1 - h)))
    
    A0 = A0 - alpha / m * np.sum((2 * arr2[:,2] - 1) * h * (1 - h))
    A1 = A1 - alpha / m * np.sum((2 * arr2[:,2] - 1) * h * (1 - h) * arr2[:,0])
    A2 = A2 - alpha / m * np.sum((2 * arr2[:,2] - 1) * h * (1 - h) * arr2[:,1])
    



plt.figure(figsize=(10, 10))
plt.subplot(2,1,1)
plt.plot(cost)
plt.xlabel("times")
plt.ylabel("cost")
plt.title("convergence of grad")



Class = np.zeros(len(arr2))
PTcount = 0
NTcount = 0
PFcount = 0
NFcount = 0
for i in range(len(arr2)):
    if h[i] > 0.5:
        Class[i] = 1
        if Class[i] == arr2[i, 2]:
            PTcount = PTcount + 1
        else:
            NTcount = NTcount + 1
    
    else:
        Class[i] = 0
        if Class[i] == arr2[i, 2]:
            PFcount = PFcount + 1
        else:
            NFcount = NFcount + 1
            
percision = PTcount / (PTcount + PFcount)
recall = PTcount / (PTcount + NFcount)
f = 2 * percision * recall / (percision + recall)
    
print("h = ({0}) + ({1})x + ({2})y)".format(A0, A1, A2))
print("PT　：　", PTcount)
print("NT　：　", NTcount)
print("PF　：　", PFcount)
print("NF　：　", NFcount)
print("F 　：　", f)


# =============================================================================
# グラフの表示（データ、分類の式）
# =============================================================================
x = np.arange(arr2[:, 0].min(), arr2[:, 0].max()+1)
y = np.zeros(len(x))
for i in range(len(x)):
    y[i] = -1/A2 * (A1 * x[i] + A0)


plt.figure(figsize=(10,10))
plt.subplot(2,1,2)
plt.scatter(arr2[0:46, 0], arr2[0:46, 1])
plt.scatter(arr2[47:127, 0], arr2[47:127, 1])
#plt.scatter(arr2[:, 0], arr2[:, 1])
plt.plot(x, y, c="black")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(arr2[:, 0].min()-1, arr2[:, 0].max()+1)
plt.ylim(arr2[:, 1].min()-1, arr2[:, 1].max()+1)
plt.title("classification")


