# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:01:57 2018

@author: yoshidako
np.polyfitを用いて回帰直線を求める
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


"""
データの取得部分
"""
#ExcelファイルをDataFrameで読み込む
input_book = pd.ExcelFile('C:/Users/aizawa/Desktop/kotaro_data.csv')
#Sheetの名前を取得
input_sheet_name = input_book.sheet_names

#DataFrameとして一つ目のsheetを読込
input_shieet_df = input_book.parse(input_sheet_name[0])
df = input_book.parse(input_sheet_name[0]) 
df_columns = df.columns.values
#df.plot(x=df_columns[0],y=df_columns[1],kind="scatter")

"""
クラスタリング部分
"""
#特徴量をXに格納
X=np.array(df.loc[:, list(df_columns[0:2])].values)

#k-means法を用いて分類問題を解く
kmeans = KMeans(n_clusters=2,init="k-means++" ).fit(X)
#データのラベルを確認する
labels = kmeans.labels_

#ラベルごとに値を再格納
X1, X2  = X[labels==0], X[labels==1]
#X3= X[labels==2]

#図でクラスタリングできているか可視化
fig = plt.figure(1) # Figureを作成
ax = fig.add_subplot(111) # Axesを作成
ax.scatter(X1[:,0],X1[:,1], label='X1')
ax.scatter(X2[:,0], X2[:,1],label='X2') 
#ax.scatter(X3[:,0],X3[:,1], label='X3')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original data')
plt.legend()

"""
回帰部分
"""
#モジュールを用いてparameterの学習
#最急降下法を用いる方法の時とパラメータが逆なので修正
theta1=np.polyfit(X1[:,0],X1[:,1],1)
theta1=theta1[::-1]
theta2=np.polyfit(X2[:,0],X2[:,1],1)
theta2=theta2[::-1]
#theta3=np.polyfit(X3[:,0],X3[:,1],1)
#theta3=theta3[::-1]


h1=theta1[1]*X1[:,0]+theta1[0]
h2=theta2[1]*X2[:,0]+theta2[0]
#h3=theta3[1]*X3[:,0]+theta3[0]


#元データに回帰直線を当てはめる図出力
"""
fig = plt.figure(3) # Figureを作成
ax = fig.add_subplot(111) # Axesを作成
ax.scatter(X1[:,0],X1[:,1], label='X1')
ax.scatter(X1[:,0], h1,label='h2') 
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original data')
plt.legend()

fig = plt.figure(4) # Figureを作成
ax = fig.add_subplot(111) # Axesを作成
ax.scatter(X2[:,0],X2[:,1], label='X1')
ax.scatter(X2[:,0], h2,label='h1') 
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original data')
plt.legend()
"""
fig = plt.figure(5) # Figureを作成
ax = fig.add_subplot(111) # Axesを作成
ax.scatter(X[:,0],X[:,1], label='X')
ax.plot([X2[X2[:,0]==np.min(X2[:,0]),0],X2[X2[:,0]==np.max(X2[:,0]),0]], [h2[X2[:,0]==np.min(X2[:,0])],h2[X2[:,0]==np.max(X2[:,0])]],label='h2') 
ax.plot([X1[X1[:,0]==np.min(X1[:,0]),0],X1[X1[:,0]==np.max(X1[:,0]),0]], [h1[X1[:,0]==np.min(X1[:,0])],h1[X1[:,0]==np.max(X1[:,0])]],label='h1')   
#ax.plot([X3[0,0],X3[-1,0]], [h3[0],h3[-1]],label='h3') 
plt.xlabel('Hight')
plt.ylabel('Weight')
plt.xlim([np.min(X[:,0])-1,np.max(X[:,0])+1])
plt.title('Japanese data')
plt.legend()

#回帰直線の出力
print("回帰直線：")
print("y={}x+{}".format(theta1[1],theta1[0]))
print("y={}x+{}\n".format(theta2[1],theta2[0]))
