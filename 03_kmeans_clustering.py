import ast
import pandas as pd
from sklearn.cluster import KMeans

#input pickle should contain two columns document and embedding
df = pd.read_pickle(r"path to document+embedding csv")
# print(len(df))
# print(df.columns)

X = [ast.literal_eval(i)[0] for i in list(df['embedding'])]
# print(type(X))
# print(np.shape(X))

# i=3
kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=0, verbose=1)
kmeans.fit(X)
labels = list(kmeans.labels_)
# print(labels)

documents = list(df['documents'])
predicted_classes = labels

out_df = pd.DataFrame()
out_df['document'] = documents
out_df['predicted_classe'] = predicted_classes
out_df.to_csv(r"path to save classes")

# Plotting

# wcss = []
# for i in range(1, 11):
#     if(i==2):break
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.show()



# kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
# pred_y = kmeans.fit_predict(X)
# print(pred_y)
# plt.scatter(X[:,0], X[:,1])
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
# plt.show()