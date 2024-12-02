import openai
import numpy as np
import hdbscan

import umap
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.colors import Normalize
from base import Framework, Population, Cluster, initialize_session, Generation
from prompts.mutation_base import get_init_archive
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import uuid

from sqlalchemy.orm import object_session
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

        session = object_session(population)

        generation = Generation(session=session, population_id=population.population_id)

        embeddings = [framework.framework_descriptor for framework in population.frameworks]
        
        mode_framework_shape = np.shape(embeddings[0])[0] 

        # Set any wrong ones to the median shape of zeros
        for i, descriptor in enumerate(embeddings):
            if not descriptor or len(descriptor) != mode_framework_shape:
                embeddings[i] = np.zeros((int(mode_framework_shape),))

        for i, descriptor in enumerate(embeddings):
            print(f"Framework {i} descriptor shape: {np.shape(descriptor)}")

        embeddings = np.array(embeddings, dtype=np.float32)
        print(f"Embeddings shape: {embeddings.shape}")


        labels = self.clusterer.fit_predict(embeddings)

        
        # get unique labels and assign them to the clusters
        unique_labels = np.unique(labels)

        if len(unique_labels) < 10:
            for framework in population.frameworks:
                cluster = Cluster(session=session, generation_id=generation.generation_id, population_id=population.population_id)
                population.clusters.append(cluster)
                framework.update(cluster_id = cluster.cluster_id)
        else:
            for label in unique_labels:
                cluster = Cluster(session=session, generation_id=generation.generation_id, population_id=population.population_id)
                population.clusters.append(cluster)
                for i, framework in enumerate(population.frameworks):
                    if labels[i] == label:
                        framework.update(cluster_id = cluster.cluster_id)

        return labels

