import openai
import numpy as np
import hdbscan

import umap
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.colors import Normalize
from base import Framework, Population, Cluster, initialize_session
from prompts.mutation_base import get_init_archive
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import logging

from dotenv import load_dotenv
load_dotenv(override=True)

class Descriptor:

    def __init__(self, model="text-embedding-3-small", output_dim=24):
        self.client = openai.Client()
        self.model = model
        self.output_dim = output_dim

    def batch_generate(self, frameworks: list[Framework]):
        with ThreadPoolExecutor(max_workers=16) as executor:
            embeddings = list(tqdm(executor.map(self.generate, frameworks), total=len(frameworks), desc="Generating embeddings"))
        return np.array(embeddings)
    
    def generate(self, framework:Framework):
        text = framework.framework_name + ": " + framework.framework_thought_process + "\n" + framework.framework_code
        response = self.client.embeddings.create(input=text, model=self.model, dimensions=self.output_dim)
        return response.data[0].embedding


class Clusterer:

    def __init__(self, min_cluster_size=3, min_samples=1, metric="euclidean"):
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)

    def cluster(self, population: Population):

        embeddings = [framework.framework_descriptor for framework in population.frameworks]
        labels = self.clusterer.fit_predict(embeddings)

        
        # get unique labels and assign them to the clusters
        unique_labels = np.unique(labels)

        if len(unique_labels) < 10:
            for framework in population.frameworks:
                cluster = Cluster()
                framework.update(cluster_id = cluster.cluster_id)
        else:
            for label in unique_labels:
                cluster = Cluster()
                for i, framework in enumerate(population.frameworks):
                    if labels[i] == label:
                        framework.update(cluster_id = cluster.cluster_id)

        return labels


class Visualizer:
    
    def __init__(self, n_components=2, random_state=42):
        self.reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        
    def plot(self, population):
        # Extract embeddings and labels
        embeddings = [framework.framework_descriptor for framework in population.frameworks]
        labels = [framework.cluster_id for framework in population.frameworks]
        
        # Map non-numeric labels to numeric values
        unique_labels = list(set(labels))
        label_to_numeric = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [label_to_numeric[label] for label in labels]

        
        # Reduce dimensions with UMAP
        reduced_embeddings = self.reducer.fit_transform(embeddings)
        
        # Create scatter plot with colors for clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1], 
            c=numeric_labels, 
            cmap='tab10',  # A colormap for discrete clusters
            s=50,  # Size of points
            alpha=0.8
        )
        
        # Add a colorbar to interpret cluster IDs
        plt.colorbar(scatter, label='Cluster ID')
        plt.title('Cluster Visualization with UMAP', fontsize=14)
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.show()




if __name__ == "__main__":

    session, Base = initialize_session()

    # Create a descriptor object
    descriptor = Descriptor()

    archive = get_init_archive()

    population = Population()   

    frameworks = []

    for framework in archive:
        frameworks.append(Framework(
            framework_name=framework['name'],
            framework_code=framework['code'],
            framework_thought_process=framework['thought'],
            framework_generation=0,
            population=population
        ))

    # Generate embeddings for the frameworks

    for framework in population.frameworks:
        framework.update(framework_descriptor = descriptor.generate(framework))

    # Create a clusterer object
    clusterer = Clusterer()

    # Cluster the embeddings
    labels = clusterer.cluster(population)

    # Create a visualizer object
    visualizer = Visualizer()



    # Plot the reduced embeddings with cluster colors
    visualizer.plot(population)