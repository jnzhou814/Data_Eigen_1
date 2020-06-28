# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler

# %%数据的预处理，标准化
data = pd.read_csv("E://Python project//L3//car_data.csv", encoding="GBK")
train_x = data[["人均GDP", "城镇人口比重", "交通工具消费价格指数", "百户拥有汽车量"]]
train_x.index = data["地区"]
sc = StandardScaler()
train_x_std = sc.fit_transform(train_x)

# %%进行PCA降维度
n = train_x.shape[1]
cov_mat = np.cov(train_x_std.T)
eigen_values, eigen_vecs = np.linalg.eig(cov_mat)
print("\nEigenvalues \n%s" % eigen_values)
total = sum(eigen_values)
var_exp = [(i/total) for i in sorted(eigen_values, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1, n+1), var_exp, align='center',
        label="individual eigen_values")
plt.step(range(1, n+1), cum_var_exp, where="mid", label="cumsum eigen_values")

# %%根据降维结果选择前两个维度
eigen_pairs = [(np.abs(eigen_values[i]), eigen_vecs[:, i])
               for i in range(len(eigen_values))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))

# %%根据选择的维度对数据进行处理，并查看SSE的分布
train_x_std_PCA = train_x_std.dot(w)
sse = []
for k in range(1, 18):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(train_x_std_PCA)
    sse.append(kmeans.inertia_)
x = range(1, 18)
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(x, sse, 'o-')
plt.show()

# %%选择5作为分类的种类
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters)
y_km = kmeans.fit_predict(train_x_std_PCA)
labels = []
markers = ['s', 'o', 'p', 'v', '*', 'x', 'D']
for i in range(n_clusters):
    print("label:", i, train_x[y_km == i].index.values)
    labels.append(train_x[y_km == i].index.values)
    plt.scatter(train_x_std_PCA[y_km == i, 0],
                train_x_std_PCA[y_km == i, 1], marker=markers[i], label=i)
plt.legend()
plt.show()

# %%使用层次聚类
model = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
y = model.fit_predict(train_x_std_PCA)
print(y)
linkage_matrix = ward(train_x_std_PCA)
dendrogram(linkage_matrix)
plt.show()
