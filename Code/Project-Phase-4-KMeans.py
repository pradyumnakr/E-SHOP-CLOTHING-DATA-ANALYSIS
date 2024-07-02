from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler

dataFrame = pd.read_csv('e-shop-clothing.csv')

dataFrame = dataFrame.sample(n=50000).reset_index(drop=True)

labelEncoder = LabelEncoder()
dataFrame['page 2 (clothing model)'] = labelEncoder.fit_transform(dataFrame['page 2 (clothing model)'])

numerical_columns = ['order', 'price']
Q1 = dataFrame[numerical_columns].quantile(0.25)
Q3 = dataFrame[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

condition = (dataFrame[numerical_columns] >= lower_bound) & (dataFrame[numerical_columns] <= upper_bound)
dataFrame = dataFrame.loc[condition['order']]
dataFrame = dataFrame.loc[condition['price']]

dataFrame = dataFrame.reset_index(drop=True)

numerical_df = dataFrame[['order','price']]
categorical_df = dataFrame[['month', 'day', 'country', 'page 1 (main category)', 'colour','location','model photography','page']]

X = pd.concat([numerical_df, categorical_df], axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

wcss = []

for k in range(1,12):
    model = KMeans(n_clusters=k, random_state=5805, verbose=2)
    model.fit(X)
    wcss.append(model.inertia_)

plt.figure(figsize=(10,6))
plt.plot(range(1,12), wcss, label='K-Means',color='blue')
plt.xticks(range(1,12))
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within Cluster Sum of Squares')
plt.legend()
plt.grid(True)
plt.show()


slc = []
for k in range(2, 12):
    kmeans = KMeans(n_clusters=k, random_state=5805, verbose=2).fit(X)
    score = silhouette_score(X, kmeans.labels_, random_state=5805)
    slc.append(score)

plt.plot(range(2,12),slc, 'bx-')
plt.xticks(range(2,12))
plt.title('Silhouette Method')
plt.xlabel('Number of clusters (k)')
plt.grid()
plt.ylabel('Silhouette Coefficient')
plt.show()

