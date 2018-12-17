import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, median,variance,stdev
from sklearn import linear_model



df1 = pd.read_csv("C:/Users/aizawa/Desktop/programing/RobustRegression/outlier_kuramoto2.csv", header=0)
#print(df1)


# =============================================================================
# 基準の相関係数
# =============================================================================
ax = np.zeros([len(df1)])
ay = np.zeros([len(df1)])
s = np.zeros([len(df1)])
sx1 = np.zeros([len(df1), len(df1)])
sy1 = np.zeros([len(df1), len(df1)])
s1 = np.zeros([len(df1), len(df1)])
sxy1 = np.zeros([len(df1), len(df1)])

for i in range(len(df1)):
    for j in range(len(df1)):
        if i==j:
            ax[j] = (df1.iloc[j,0] - df1.iloc[:,0].mean())**2
            ay[j] = (df1.iloc[j,1] - df1.iloc[:,1].mean())**2
            s[j] = (df1.iloc[j,0] - df1.iloc[:,0].mean()) * (df1.iloc[j,1] - df1.iloc[:,1].mean())
        else:
            sx1[j, i] = (df1.iloc[j,0] - df1.iloc[:,0].mean())**2
            sy1[j, i] = (df1.iloc[j,1] - df1.iloc[:,1].mean())**2
            s1[j, i] = (df1.iloc[j,0] - df1.iloc[:,0].mean()) * (df1.iloc[j,1] - df1.iloc[:,1].mean())


ax = ax.sum() / len(ax)
ay = ay.sum() / len(ay)
axy = s.sum() / len(s)
sxy = axy/np.sqrt(ax*ay)
print(sxy)


ax1 = np.zeros([len(df1)])
ay1 = np.zeros([len(df1)])
axy1 = np.zeros([len(df1)])
sxy1 = np.zeros([len(df1)])
e1 = np.zeros([len(df1)])

 
for i in range(len(df1)):
    ax1[i] = sx1[:, i].sum() / len(sx1)
    ay1[i] = sy1[:, i].sum() / len(sy1)
    axy1[i] = s1[:, i].sum() / len(s1)
    sxy1[i] = abs(axy1[i]/np.sqrt(ax1[i]*ay1[i]))
    e1[i] = abs(sxy1[i] / sxy)

ex1 = np.array(df1.iloc[:, 0])
ex2 = np.array(df1.iloc[:, 1])
e = np.c_[ex1, e1]
e = np.c_[e, ex2]
#
#print(ex)
##print(e)
#plt.subplot(4,1,1)
#plt.figure(figsize=(10,5))
#plt.scatter(e[:, 0], e[:, 1])
#plt.scatter(df1.iloc[:,0], sxy1)

n =0.36
e2 = np.array([0, median(e[:, 1]), median(e[:, 2])])
for i in range(len(e)):
    if e[i, 1] < sxy/n:
        e2 = np.vstack([e2, e[i]])
#        print(e2)

print(e2.shape)

#plt.subplot(2,1,1)
#plt.figure(figsize=(13,5))
#plt.scatter(e2[1:, 0], e2[1:, 2])
#plt.show

        
#theta1=np.polyfit(e2[:,0], e2[:,2], 1)
#theta1=theta1[::-1]
#h1=theta1[1]*e2[:,0]+theta1[0]


clf = linear_model.LinearRegression()
X = e2[1:, 0].reshape(1, -1)
Y = e2[1:, 2].reshape(1, -1)
clf.fit(X, Y)

print(clf.coef_)
print(clf.intercept_)

#line_X = np.array([e2[1:, 0].min(), e2[1:, 0].max()])
#line_Y = clf.predict()
#

#
#plt.subplot(4,1,2)
#plt.figure(figsize=(10,5))
plt.scatter(df1.iloc[:, 0], df1.iloc[:, 1], c="red")
plt.scatter(e2[1:, 0], e2[1:, 2])
plt.plot(X[0,:], clf.predict(X)[0,:])
#
#n = 10
#e3 = np.array([1,2] )
#a = np.zeros([n, 2])
#for j in range(1, n):
#    for i in range(len(e)):
#        if e[i, 1] < sxy/j:
#            e3 = np.vstack([e3, e[i]])
#    a[j, 0] = j
#    a[j, 1] = e3.shape[0]
#
#plt.subplot(2,1,2)
#plt.figure(figsize=(13,5))
#plt.scatter(a[:, 0], a[:, 1])    
#plt.show
#        
        