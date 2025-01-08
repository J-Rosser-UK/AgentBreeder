import openai
import numpy as np
import hdbscan
from base import System, Population, Cluster, Generation

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from sqlalchemy.orm import object_session

from dotenv import load_dotenv

load_dotenv(override=True)


class Descriptor:

    def __init__(self, model="text-embedding-3-small", output_dim=12):
        """
        Initializes the Descriptor class.

        Args:
            model (str): The name of the OpenAI embedding model to use for generating embeddings.
            output_dim (int): The dimensionality of the output embeddings.
        """
        self.client = openai.Client()
        self.model = model
        self.output_dim = output_dim

    def batch_generate(self, systems: list[System]):
        """
        Generates embeddings for a batch of systems using threading.

        Args:
            systems (list[System]): A list of system objects for which embeddings will be generated.

        Returns:
            np.ndarray: A NumPy array containing the embeddings for all systems in the batch.
        """
        with ThreadPoolExecutor(max_workers=16) as executor:
            embeddings = list(
                tqdm(
                    executor.map(self.generate, systems),
                    total=len(systems),
                    desc="Generating embeddings",
                )
            )
        return np.array(embeddings)

    def generate(self, system: System):
        """
        Generates an embedding for a single system.

        Args:
            system (System): The system object for which the embedding will be generated.

        Returns:
            list[float]: The embedding vector for the given system.
        """
        text = (
            system.system_name
            + ": "
            + system.system_thought_process
            + "\n"
            + system.system_code
        )
        response = self.client.embeddings.create(
            input=text, model=self.model, dimensions=self.output_dim
        )
        return response.data[0].embedding


class Clusterer:

    def __init__(self, min_cluster_size=3, min_samples=1, metric="euclidean"):
        """
        Initializes the Clusterer class.

        Args:
            min_cluster_size (int): Minimum size of a cluster.
            min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.
            metric (str): The distance metric to use for clustering.
        """

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric
        )

    def cluster(self, population: Population):
        """
        Clusters systems in a population based on their embeddings.

        Args:
            population (Population): The population object containing systems to cluster.

        Returns:
            np.ndarray: An array of cluster labels for the multi-agent systems in the population.
        """

        session = object_session(population)

        generation = Generation(session=session, population_id=population.population_id)

        embeddings = [system.system_descriptor for system in population.systems]

        mode_system_shape = np.shape(embeddings[0])[0]

        # Set any wrong ones to the median shape of zeros
        for i, descriptor in enumerate(embeddings):
            if not descriptor or len(descriptor) != mode_system_shape:
                embeddings[i] = np.zeros((int(mode_system_shape),))

        embeddings = np.array(embeddings, dtype=np.float32)
        # print(f"Embeddings shape: {embeddings.shape}")

        labels = self.clusterer.fit_predict(embeddings)

        # get unique labels and assign them to the clusters
        unique_labels = np.unique(labels)

        print("Number of unique clusters: ", len(unique_labels))

        if len(embeddings) < 10:
            for system in population.systems:
                cluster = Cluster(
                    session=session,
                    generation_id=generation.generation_id,
                    population_id=population.population_id,
                )
                population.clusters.append(cluster)
                system.update(cluster_id=cluster.cluster_id)
        else:
            for label in unique_labels:
                cluster = Cluster(
                    session=session,
                    generation_id=generation.generation_id,
                    population_id=population.population_id,
                )
                population.clusters.append(cluster)
                for i, system in enumerate(population.systems):
                    if labels[i] == label:
                        system.update(cluster_id=cluster.cluster_id)

        return labels
