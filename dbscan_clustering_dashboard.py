import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to perform DBSCAN clustering
def cluster_data(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
    return clusters

# Function to calculate silhouette score
def calculate_silhouette(data, clusters):
    noise = clusters == -1
    valid_clusters = clusters[~noise]
    valid_points = data[~noise]
    if len(np.unique(valid_clusters)) > 1:
        sil_score = silhouette_score(valid_points, valid_clusters)
        return sil_score
    else:
        return None

# Function to perform t-SNE and create a scatter plot
def plot_tsne(data, clusters):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_results = tsne.fit_transform(data)
    noise = clusters == -1
    fig, ax = plt.subplots()
    ax.scatter(tsne_results[noise, 0], tsne_results[noise, 1], c='gray', s=50, label='Noise')
    scatter = ax.scatter(tsne_results[~noise, 0], tsne_results[~noise, 1], c=clusters[~noise], cmap='viridis', s=50)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    return fig

# Streamlit interface
st.title("DBSCAN Clustering Dashboard")

# Upload data file
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)

    # Inputs for DBSCAN
    eps = st.slider("Select eps", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    min_samples = st.slider("Select min_samples", min_value=1, max_value=50, value=5, step=1)

    # Button to perform clustering
    if st.button("Cluster"):
        with st.spinner('Clustering data...'):
            clusters = cluster_data(data, eps, min_samples)
            silhouette = calculate_silhouette(data, clusters)
            fig = plot_tsne(data, clusters)
            
            # Display results
            if silhouette:
                st.success(f"Silhouette Score: {silhouette:.2f}")
            else:
                st.info("Silhouette score is not applicable due to the number of clusters.")
            st.pyplot(fig)
