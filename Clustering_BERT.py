import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


adjective_embeddings = pd.read_csv('/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/3. Semester/Research Project/adjectives_embeddings.csv')
adjective_embeddings['embedding'] = adjective_embeddings['embedding'].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=' ')
)

#***Elbow: determine optimal amount of clusters***

# Elbow method implementation
embedding_matrix = np.vstack(adjective_embeddings['embedding'].values)
'''inertia = []
k_range = range(1, 16)  # Try from 1 to 10 clusters

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embedding_matrix)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show() '''

#***Clustering***
optimal_k = 7  
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(embedding_matrix)

adjective_embeddings['cluster'] = clusters
adjective_embeddings.to_csv('adjective_clusters', index=False)


#***Where do the embeddings belong***
she_similarities = pd.read_csv('she_cosine_similarities.csv')
he_similarities = pd.read_csv('he_cosine_similarities.csv')

she_similarities['embedding'] = she_similarities['embedding'].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=' ')
)
he_similarities['embedding'] = he_similarities['embedding'].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=' ')
)

she_similarities['cluster'] = she_similarities['embedding'].apply(lambda embedding: kmeans.predict([embedding])[0])
he_similarities['cluster'] = he_similarities['embedding'].apply(lambda embedding: kmeans.predict([embedding])[0])

she_similarities.to_csv('she_with_clusters.csv', index=False)
he_similarities.to_csv('he_with_clusters.csv', index=False)



