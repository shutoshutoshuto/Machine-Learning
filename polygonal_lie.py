import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import math

#df1 = pd.read_csv("C:/Users/aizawa/Desktop/dataset.csv", header=0)
df2 = pd.read_csv("C:/Users/aizawa/Desktop/programing/LinerRegression/polygonal_line1.csv", header=0)

x = 5
a = np.zeros([len(df2)-int(round(len(df2)/x))])
b = np.zeros([int(round(len(df2)/x))])
c = np.zeros([len(df2)-1])

for i in range(len(df2)-int(round(len(df2)/x))):
    for j in range(int(round(len(df2)/x))):
        b[j] = (df2.iloc[i+j, 1] - df2.iloc[i+j+1, 1]) / (df2.iloc[i+j, 0] - df2.iloc[i+j+1, 0])
    a[i] = abs(sum(b)/len(b))

#print(a)
plt.subplot(2,1,1)
plt.plot(a)


c = a.max()
for i in range(len(a)):
    if c > a[i]:
        c = a[i]
        d = i

print(c,d)

arr = np.array(df2)
#print(arr)

theta1=np.polyfit(arr[0:d,0],arr[0:d,1],1)
theta1=theta1[::-1]
theta2=np.polyfit(arr[d:,0],arr[d:,1],1)
theta2=theta2[::-1]

#print(theta1)
#print(theta2)

h1=theta1[1]*arr[0:d,0]+theta1[0]
h2=theta2[1]*arr[d:,0]+theta2[0]

print(h1)
print(h2)
#
plt.subplot(2,1,2)
plt.scatter(arr[:, 0], arr[:, 1])
plt.plot(h1)
plt.plot(h2)


#
#for i in range(len(df1)-1):
#    c[i] = (df1.iloc[i, 1] - df1.iloc[i+1, 1]) / (df1.iloc[i, 0] - df1.iloc[i+1, 0])
#
#print(c)
#plt.plot(c)

#y=5
#for i in range(len(df1)):
#    for j in range(y-1):
#        if int(round(len(df1)*j/y)) <= i <= int(round(len(df1)*(j+1)/y)):
#            c = np.linalg.pinv(df1.iloc[:, 0].T * df1.iloc[:, 0]) * df1.iloc[:, 0].T * df1.iloc[: 1]
#            lmLR = linear_model.LinearRegression(fit_intercept=True,
#                                             normalize=False,
#                                             copy_X=True,
#                                             n_jobs=1).fit(df1[i, 0], df1[i, 1])
#        

