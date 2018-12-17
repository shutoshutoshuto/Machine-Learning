import numpy as np
import matplotlib.pyplot as plt

arrx1 = np.arange(1, 11, 1)
arry1 = np.array([2,4,6,9,9,12,13,15,18,21])
arr1 = np.c_[arrx1, arry1]


# =============================================================================
# 正規方程式
# =============================================================================
theta1 = np.zeros([2, 1])
arrx2 = np.c_[np.ones([10, 1]), arrx1]
theta1 = np.linalg.pinv(arrx2.T.dot(arrx2)).dot(arrx2.T).dot(arry1)


h1 = np.zeros([len(arrx1), 1])
for i in range(len(arrx1)):
    h1[i] = theta1[0] + theta1[1] * arrx1[i]


R1 = np.sqrt(np.sum((h1[:] - np.mean(arry1))**2) / np.sum((arry1[:] - np.mean(arry1))**2))
print("相関係数（正規方程式）　　　  　　　", R1)




# =============================================================================
# 直線フィッティングの式
# =============================================================================
theta2 = np.zeros([2, 1])
theta2[1] = np.cov(arr1, rowvar=0, bias=1)[0, 1] / np.var(arr1[:, 0])
theta2[0] = np.mean(arr1[:, 1]) - theta2[1] * np.mean(arr1[:, 0])


h2 = np.zeros([len(arr1[:, 0]), 1])
for i in range(len(arrx1)):
    h2[i] = theta2[0] + theta2[1] * arrx1[i]


R2 = np.sqrt(np.sum((h2[:] - np.mean(arr1[:, 1]))**2) / np.sum((arr1[:, 1] - np.mean(arr1[:, 1]))**2))
print("相関係数（直線フィッティングの式）　　", R2)




plt.scatter(arrx1, arry1, label="data")
plt.plot(arrx1, h1, label="Normal equation")
plt.plot(arrx1, h2, label="Liner fitting")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Liner Regression")
plt.legend(loc="lower right")
