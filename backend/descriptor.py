import openai
import numpy as np
import hdbscan

import umap
import matplotlib.pyplot as plt
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
    def __init__(self, n_components=5, random_state=42):
        self.reducer = umap.UMAP(n_components=n_components, random_state=random_state)



    def plot(self, embeddings, labels):
        
        # Normalize RGB values between 0 and 1
        reduced_embeddings = self.reducer.fit_transform(embeddings)

        # Step 2: Separate the reduced dimensions into RGB and spatial (x, y)
        rgb_values = reduced_embeddings[:, :3]  # First three dimensions for RGB
        xy_values = reduced_embeddings[:, 3:5]  # Last two dimensions for x and y

        # Normalize RGB values between 0 and 1
        rgb_normalized = (rgb_values - np.min(rgb_values, axis=0)) / (np.ptp(rgb_values, axis=0))

        # Step 3: Plot the points
        plt.figure(figsize=(8, 8))
        plt.scatter(xy_values[:, 0], xy_values[:, 1], c=rgb_normalized, s=50, alpha=0.7)
        plt.title("UMAP Dimensionality Reduction with RGB and Spatial Visualization")
        plt.xlabel("UMAP Dimension 1 (x)")
        plt.ylabel("UMAP Dimension 2 (y)")
        plt.colorbar(label="Cluster RGB Intensity (Normalized)")
        plt.grid(True)
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

    # Get the embeddings for the frameworks
    embeddings = [framework.framework_descriptor for framework in population.frameworks]


    # Plot the reduced embeddings with cluster colors
    visualizer.plot(embeddings, labels)