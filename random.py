import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#df1 = pd.read_csv("C:/Users/aizawa/Desktop/programing/LogisticRegression/aizawa.csv", header = 0)
df1 = pd.read_csv("C:/Users/aizawa/Desktop/programing/2018 1101_zirou/features.csv", header = 0)
arr1 = np.array(df1)

#
#for i in range(arr1.shape[0]):
#    arr1[i,5] = -1
#
#arr2 = np.r_[arr1[:,0:3], arr1[:,3:6]]


t = 10000
countmax = 0
opa = 0
opb = 0
opc = 0
base = 100
mina = -1 * base + 1
maxa = 1 * base + 1


arr2[:, 0] = (arr1[:, 0] - np.mean(arr1[:, 0])) / np.std(arr1[:, 0])
arr2[:, 1] = (arr1[:, 1] - np.mean(arr1[:, 1])) / np.std(arr1[:, 1])


for j in range(t):
    a = np.random.uniform(mina,maxa)
    b = np.random.uniform(mina,maxa)
    c = np.random.uniform(mina,maxa)
    Class = np.zeros(len(arr2))
    PTcount = 0
    NTcount = 0
    PFcount = 0
    NFcount = 0
    
    for i in range(len(arr2)):
        h = a * arr2[i, 0] + b * arr2[i, 1] + c
        if h > 0:
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
            
    
    if (PTcount + PFcount) > countmax:
        countmax = PTcount + PFcount
        opa = a
        opb = b
        opc = c
    
    if countmax == len(arr2):
        break

Precision = PTcount / (PTcount + PFcount)
Recall = PTcount / (PTcount + NTcount)
F = 2 * Precision * Recall / (Precision + Recall)

print("h = ({0})x + ({1})y + ({2})".format(opa, opb, opc))
print("PT　：　", PTcount)
print("NT　：　", NTcount)
print("PF　：　", PFcount)
print("NF　：　", NFcount)
print("F 　：　", F)


oph = np.zeros(len(arr2))
ophc = np.zeros(len(arr2))
oph = opa*arr2[:, 0]+opb*arr2[:, 1]+opc
for i in range(len(oph)):
    if oph[i] > 0:
        ophc[i] = 1
    else:
        ophc[i] = -1


x = np.arange(arr2[:, 0].min(), arr2[:, 0].max())
y = np.zeros(len(x))
for i in range(len(x)):
    y[i] = -1/opb * (opa * x[i] + opc)



plt.subplot(2,1,2)
#plt.scatter(arr1[:, 0], arr1[:, 1])
#plt.scatter(arr1[:, 3], arr1[:, 4])
#plt.plot(x, y, c="black")
plt.scatter(arr2[0:46, 0], arr2[0:46, 1])
plt.scatter(arr2[47:127, 0], arr2[47:127, 1])
plt.plot(x, y, c="black")

#
#R = np.sqrt(np.sum((ophc[:] - arr2[:,2].mean())**2) / np.sum((arr2[:,2] - arr2[:,2].mean())**2))
#print("相関係数　：　", R)