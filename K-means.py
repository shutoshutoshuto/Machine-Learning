import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# データ読み込み
urusa_df = pd.read_csv("C:/Users/aizawa/Documents/WinPython-64bit-3.6.3.0Qt5/scripts/20180719/Switch_kakaku.csv")


# 列削除
del(urusa_df['No'])
del(urusa_df['model'])
# del(urusa_df['design'])
# del(urusa_df['soft quality'])
# del(urusa_df['operational feeling'])
# del(urusa_df['image quality'])
# del(urusa_df['size'])
# del(urusa_df['scalability'])
del(urusa_df['review'])


# 項目ごと二重配列
urusa_array = np.array([
                       # urusa_df['No'].tolist(),
                       # urusa_df['model'].tolist(),
                       urusa_df['design'].tolist(),
                       urusa_df['soft quality'].tolist(),
                       urusa_df['operational feeling'].tolist(),
                       urusa_df['image quality'].tolist(),
                       urusa_df['size'].tolist(),
                       urusa_df['scalability'].tolist(),
                       # urusa_df['review'].tolist(),
                       ])



# サンプルごと配列整理
urusa_array = urusa_array.T


# エルボー分析の表示
distortions = []
for i in range(1, 17):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=20,
                max_iter=600,
                random_state=0)
    km.fit(urusa_array)
    distortions.append(km.inertia_)

plt.plot(range(1,17),
         distortions,
         marker='D')
plt.title('Elbow analys')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion(SSE)')
plt.savefig("cluster_elbow.png")
plt.show()



# クラスタ数指定
print('What is the number of clusters?')
n=input()




# k-means
km = KMeans(n_clusters=int(n),
            init='k-means++',
            n_init=20,
            max_iter=600,
            random_state=0).fit_predict(urusa_array)



# cust_dfにcluster_id列を作り、kmを代入
urusa_df['cluster_id']=km



# クラスタごとの個数
urusa_df['cluster_id'].value_counts()



# クラスタの詳細
import csv

f = open('cluser_value.csv','w')
csvwriter = csv.writer(f)
cluster_value = []

for i in range(int(n)):
        cluster_value.append(i)
        value = urusa_df[urusa_df['cluster_id']==i].mean()
        # print(value)
        cluster_value.append(value)

csvwriter.writerow(cluster_value)
f.close()




# シルエット分析
from sklearn.metrics import silhouette_samples
from matplotlib import cm

km = KMeans(n_clusters=int(n),
            init='k-means++',
            n_init=20,
            max_iter=600,
            random_state=0).fit_predict(urusa_array)

cluster_labels = np.unique(km)
s_clusters=cluster_labels.shape[0]

shilhouette = silhouette_samples(urusa_array,km,metric='euclidean')

y_lower=0
y_upper=0
y_thick=[]

for i in range(int(s_clusters)):
        i_shilhouette =shilhouette[km==i]
        i_shilhouette.sort()
        y_upper += len(i_shilhouette)
        plt.barh(range(y_lower,y_upper),
                 i_shilhouette,
                 height = 1.0,
                 color = cm.jet(float(i)/s_clusters))
        y_thick.append((y_lower + y_upper)/2)
        y_lower += len(i_shilhouette)

shilhouette_ave = np.mean(shilhouette)
plt.title('Shilhouette analys')
plt.axvline(shilhouette_ave, color = 'k', linestyle ='--')
plt.yticks(y_thick,cluster_labels +1)
plt.xlabel('Shilhouette coefficient')
plt.ylabel('Cluster')
plt.savefig("cluster_Silhouette.png")
plt.show()




# グラフ表示
clusterinfo = pd.DataFrame()
for i in range(int(n)):
    clusterinfo['cluster' + str(i)] = urusa_df[urusa_df['cluster_id'] == i].mean()
clusterinfo = clusterinfo.drop('cluster_id')
print(clusterinfo)
print(clusterinfo.T)
my_plot = clusterinfo.T.plot(kind='bar',
                             stacked=True,
                             title='Value of ' +str(n)+ ' Clusters')
my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=0)

# for y in range(clusterinfo.shape[0]):
#     for x in range(clusterinfo.shape[1]):
#         plt.text(int(n), y, clusterinfo, horizontalalignment='center', verticalalignment='center')

# plt.text(ha='center', va='bottom')
plt.ylabel('Level of satisfaction')
plt.savefig('cluster_analys.png')
plt.show()
