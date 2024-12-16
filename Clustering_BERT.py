import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
from wordcloud import WordCloud


adjective_embeddings = pd.read_csv('/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/3. Semester/Research Project/adjectives_embeddings.csv')
adjective_embeddings['embedding'] = adjective_embeddings['embedding'].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=' ')
)

noun_embeddings = pd.read_csv('/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/3. Semester/Research Project/noun_embeddings.csv')
noun_embeddings['embedding'] = noun_embeddings['embedding'].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=' ')
)

'''verb_embeddings = pd.read_csv('/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/3. Semester/Research Project/verb_embeddings.csv')
verb_embeddings['embedding'] = verb_embeddings['embedding'].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=' ')
)'''



#*****Elbow + Silhouette method to determine optimal k*****
combined_embeddings = pd.concat([adjective_embeddings[['word', 'embedding']], noun_embeddings[['word', 'embedding']]])
embedding_matrix = np.vstack(combined_embeddings['embedding'].values)
'''inertia = []
silhouette_scores = []
k_range = range(2, 25) # chosen as a result of 200 data points avoid overfitting => less meaningfull clustes.  
#K range increased to 20 due to more data points after the addition of nouns
# 12/12/2024:
# K range increased to 25 to see if optimal k changes - optimal k is decided to be 10 from elbow point.
# Increasing k to 20 for fun, results: some are more fine grained, while other are messier/messy.

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embedding_matrix)
    inertia.append(kmeans.inertia_)

    score = silhouette_score(embedding_matrix, kmeans.labels_)
    silhouette_scores.append(score)


plt.subplot(1, 2, 1)  # Subplot for elbow method
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)

# Plot Silhouette Scores
plt.subplot(1, 2, 2)  # Subplot for silhouette method
plt.plot(k_range, silhouette_scores, marker='o', linestyle='--', color='green')
plt.title('Silhouette Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)

# Show the plots side by side
plt.tight_layout()
plt.show()'''

#***Clustering***
#For nouns + adjectives - 8, 6 dog decimaler denne gang.
#For adjectives - 8, 6.
optimal_k = 8
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(embedding_matrix)

combined_embeddings['cluster'] = clusters
combined_embeddings.to_csv('combined_clusters.csv', index=False)

#****Visualization with word clouds****
'''cluster_words = defaultdict(list)
for _, row in combined_embeddings.iterrows():
    cluster_words[row['cluster']].append(row['word'])

# Ensure unique words in each cluster
for cluster_id in cluster_words:
    cluster_words[cluster_id] = list(set(cluster_words[cluster_id]))

# Step 4: Generate Word Clouds for Each Cluster
for cluster_id, words in cluster_words.items():
    wordcloud = WordCloud(
        background_color="white",  
        width=800, height=400,    
        max_words=150,            
        colormap="viridis"        
    ).generate(" ".join(words))
    
    # Plot the Word Cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"Cluster {cluster_id}")
    plt.axis("off")  # Hide axes for better visualization
    plt.show()'''

#***Where do the embeddings belong***
she_similarities = pd.read_csv('she_cosine_similarities_ext.csv')
he_similarities = pd.read_csv('he_cosine_similarities_ext.csv')

she_similarities['embedding'] = she_similarities['embedding'].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=' ')
)
he_similarities['embedding'] = he_similarities['embedding'].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=' ')
)

she_similarities['cluster'] = she_similarities['embedding'].apply(lambda embedding: kmeans.predict([embedding])[0])
he_similarities['cluster'] = he_similarities['embedding'].apply(lambda embedding: kmeans.predict([embedding])[0])

she_similarities_without_embedding = she_similarities.drop(columns=['embedding'])
he_similarities_without_embedding = he_similarities.drop(columns=['embedding'])
she_similarities_without_embedding.to_csv('she_clusters_without_embeddings_ext.csv', index=False)
he_similarities_without_embedding.to_csv('he_clusters_without_embeddings_ext.csv', index=False)



