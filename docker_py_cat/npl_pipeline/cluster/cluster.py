#CLUSTERING
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from collections import Counter
import re

embeddings = np.loadtxt("/app/data/embeddings.csv", delimiter=",")
df = pd.read_csv("/app/data/processed_data.csv")
embeddings_test = np.loadtxt("/app/data/embeddings_test.csv", delimiter=",")
df_test = pd.read_csv("/app/data/processed_data_test.csv")

embeddings_cat = pd.read_csv("/app/data/embedded_keywords.csv", delimiter=",")

def tokenize(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    return tokens

embeddings_cat['Embeddings'] = embeddings_cat['Embeddings'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

initial_cluster_centers = np.array(embeddings_cat["Embeddings"].tolist())
categories = embeddings_cat["Category"].tolist()

kmeans = KMeans(n_clusters=len(embeddings_cat), init=initial_cluster_centers, n_init=1, max_iter=300, random_state=42)
kmeans.fit(embeddings)

def map_clusters_to_categories(kmeans, initial_cluster_centers, categories):
    cluster_mapping = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        distances = np.linalg.norm(initial_cluster_centers - center, axis=1)
        closest_center_index = np.argmin(distances)
        cluster_mapping[i] = categories[closest_center_index]
    return cluster_mapping

cluster_mapping = map_clusters_to_categories(kmeans, initial_cluster_centers, categories)

df['Cluster_KMeans'] = kmeans.labels_
df['Category_KMeans'] = df['Cluster_KMeans'].map(lambda x: f"{x} ({cluster_mapping[x]})")

print(df)

dbscan = DBSCAN(eps=20.0, min_samples=5)
dbscan_labels = dbscan.fit_predict(embeddings)

df['Cluster_DBSCAN'] = dbscan_labels

df.to_csv("/app/data/clustering_results.csv", index=False)



def analyze_clusters(df, text_column='Lemmas', cluster_column='Cluster_KMeans'):
    clusters = df[cluster_column].unique()
    cluster_keywords = {}

    for cluster in clusters:
        cluster_data = df[df[cluster_column] == cluster]
        all_words = []

        for text in cluster_data[text_column]:
            tokens = tokenize(str(text))
            all_words.extend(tokens)

        word_freq = Counter(all_words)
        cluster_keywords[cluster] = word_freq.most_common(10)

    return cluster_keywords

cluster_keywords = analyze_clusters(df)
output_path = "/app/data/cluster_keywords.txt"

with open(output_path, "w") as f:
    for cluster, keywords in cluster_keywords.items():
        f.write(f"Cluster_KMeans {cluster} ({cluster_mapping[cluster]}):\n")
        for word, freq in keywords:
            f.write(f"  {word}: {freq}\n")
        f.write("\n")

print(f"Cluster analysis for KMeans saved to {output_path}")

#Revise DBSCANS clusters in comparison 

def analyze_dbscan_clusters(df, text_column='Lemmas', cluster_column= 'Cluster_DBSCAN'):
    clusters = df[cluster_column].unique()
    cluster_keywords = {}

    for cluster in clusters:
        cluster_data = df[df[cluster_column] == cluster]
        all_words = []

        for text in cluster_data[text_column]:
            tokens = tokenize(str(text))
            all_words.extend(tokens)

        word_frequency = Counter(all_words)
        cluster_keywords[cluster] = word_frequency.most_common(10)

    return cluster_keywords

cluster_keywords_dbscan = analyze_dbscan_clusters(df)
output_path_dbscan = "/app/data/cluster_keywords_dbscan.txt"

with open(output_path_dbscan, "w") as f:
    for cluster, keywords in cluster_keywords_dbscan.items():
        f.write(f"Cluster {cluster}:\n")
        for word, freq in keywords:
            f.write(f"  {word}: {freq}\n")
        f.write("\n")

print(f"Cluster analysis for DBSCAN saved to {output_path_dbscan}")
cluster_labels_test = kmeans.predict(embeddings_test)
df_test['Cluster_KMeans'] = cluster_labels_test
df_test['Category'] = df_test['Cluster_KMeans'].map(lambda x: f"{x} ({cluster_mapping[x]})")

df_test.to_csv("/app/data/clustering_results_test.csv", index=False)

cluster_keywords_test = analyze_clusters(df_test)

output_path_test = "/app/data/cluster_keywords_testing.txt"
with open(output_path_test, "w") as f:
    for cluster, keywords in cluster_keywords_test.items():
        f.write(f"Cluster_KMeans {cluster} ({cluster_mapping[cluster]}):\n")
        for word, freq in keywords:
            f.write(f"  {word}: {freq}\n")
        f.write("\n")

print(f"Cluster analysis for testing data saved to {output_path_test}")
