import openai
import numpy as np


import numpy as np
from sqlalchemy.orm import object_session
from sklearn.cluster import AgglomerativeClustering

from base import Cluster, Generation


from sqlalchemy.orm import object_session

from dotenv import load_dotenv

load_dotenv(override=True)


class Clusterer:
    def __init__(self, n_clusters=None, metric="euclidean", linkage="ward"):
        """
        Initializes the Clusterer class with Agglomerative Clustering (pure hierarchical clustering).

        Args:
            n_clusters (int or None): Number of clusters to find. If None,
                                      the algorithm will not stop until each cluster contains a single sample
                                      (you may specify distance_threshold instead).
            metric (str): The distance metric to use for clustering (e.g., 'euclidean', 'manhattan', etc.).
            linkage (str): Which linkage criterion to use ('ward', 'complete', 'average', 'single').
        """

        # Note: 'ward' only works with 'euclidean'. If you need another metric, change linkage accordingly.
        # For example, if you want to use 'manhattan', you cannot use 'ward'; you might use 'complete' or 'average'.

        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage

        # If you want to use a distance threshold instead of a fixed number of clusters, you can do:
        #
        self.clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.7,  # example threshold
            # affinity=self.metric,
            linkage=self.linkage,
        )
        #
        # # But for simplicity, we'll assume a fixed n_clusters or None with no threshold:
        # self.clusterer = AgglomerativeClustering(
        #     n_clusters=self.n_clusters, affinity=self.metric, linkage=self.linkage
        # )

    def cluster(self, population):
        """
        Clusters systems in a population based on their embeddings using pure hierarchical clustering.

        Args:
            population (Population): The population object containing systems to cluster.

        Returns:
            np.ndarray: An array of cluster labels for the multi-agent systems in the population.
        """

        session = object_session(population)

        # Create a new generation
        generation = Generation(session=session, population_id=population.population_id)

        # Extract embeddings from population systems
        embeddings = [system.system_descriptor for system in population.systems]

        # Make sure each embedding has the same shape; replace mismatches with zeros
        mode_system_shape = np.shape(embeddings[0])[0]
        for i, descriptor in enumerate(embeddings):
            if not descriptor or len(descriptor) != mode_system_shape:
                embeddings[i] = np.zeros((int(mode_system_shape),))

        # Convert to numpy array
        embeddings = np.array(embeddings, dtype=np.float32)

        # Perform hierarchical clustering
        labels = self.clusterer.fit_predict(embeddings)

        # Get unique labels (clusters)
        unique_labels = np.unique(labels)
        print("Number of unique clusters: ", len(unique_labels))

        # Example: if you want to handle small populations differently
        if len(embeddings) < 10:
            # Put each system in its own cluster
            for system in population.systems:
                cluster = Cluster(
                    session=session,
                    generation_id=generation.generation_id,
                    population_id=population.population_id,
                )
                population.clusters.append(cluster)
                system.update(cluster_id=cluster.cluster_id)

        else:
            # Create one Cluster object per unique label
            for label in unique_labels:
                cluster = Cluster(
                    session=session,
                    generation_id=generation.generation_id,
                    population_id=population.population_id,
                )
                population.clusters.append(cluster)

                # Assign systems with the current label to this cluster
                for i, system in enumerate(population.systems):
                    if labels[i] == label:
                        system.update(cluster_id=cluster.cluster_id)

        return labels
