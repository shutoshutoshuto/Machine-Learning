import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# データセット
# =============================================================================
df1 = pd.read_csv("C:/Users/aizawa/Desktop/programing/LogisticRegression/class_kuramoto2.csv", header = 0)
#df1 = pd.read_csv("C:/Users/aizawa/Desktop/programing/2018 1101_zirou/features.csv", header = 0)
arr2 = np.array(df1)


# =============================================================================
# 1,2の2値を1,-1に変換
# =============================================================================
#for i in range(arr1.shape[0]):
#    arr1[i,5] = -1
#
#arr2 = np.r_[arr1[:,0:3], arr1[:,3:6]]



# =============================================================================
# 最急降下法
# =============================================================================
m = arr2.shape[0]
cost=[]    
alpha = 0.0001

a = 6
b = 0
c = 3
t = 50000

for i in range(t):
    h = a * arr2[:, 0] + b * arr2[:, 1] +c
    cost.append((np.sum((h - arr2[:, 2])**2)))                  #hとラベルの差の2乗
    a = a - alpha/m * np.sum((h - arr2[:, 2]) * arr2[:, 0])     #aで微分
    b = b - alpha/m * np.sum((h - arr2[:, 2]) * arr2[:, 1])     #bで微分
    c = c - alpha/m * np.sum((h - arr2[:, 2]))                  #cで微分


print("h = ({0})x + ({1})y + ({2})".format(a, b, c))

plt.figure(figsize=(10, 10))
plt.subplot(2,1,1)
plt.plot(cost)
plt.xlabel("times")
plt.ylabel("cost")
plt.title("convergence of grad")


# =============================================================================
# 正答率の計算
# =============================================================================
count = 0
for i in range(len(arr2)):
    h = a * arr2[i, 0] + b * arr2[i, 1] +c
    if h > 0 and arr2[i, 2] == 1:
        count = count + 1
    if h < 0 and arr2[i, 2] == -1:
        count = count + 1

print("score　：　", count/len(arr2))



# =============================================================================
# グラフの表示（データ、分類の式）
# =============================================================================
x = np.arange(arr2[:, 0].min(), arr2[:, 0].max()+1)
y = np.zeros(len(x))
for i in range(len(x)):
    y[i] = -1/b * (a * x[i] + c)


plt.figure(figsize=(10,10))
plt.subplot(2,1,2)
plt.scatter(arr2[0:4, 0], arr2[0:4, 1])
plt.scatter(arr2[4:8, 0], arr2[4:8, 1])
#plt.scatter(arr2[:, 0], arr2[:, 1])
plt.plot(x, y, c="black")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("classification")


