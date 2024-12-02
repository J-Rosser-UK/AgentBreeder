import argparse
import random

import logging
from base import initialize_session, Population, Framework

from umap import UMAP
import numpy as np

import plotly.express as px
import numpy as np
import pandas as pd
from umap import UMAP

class Visualizer:

        
    def plot(self, session, population_id):
        """
        Plots the clusters in 3D for a given population using UMAP for dimensionality reduction.
        Hovering over a point displays the framework name, framework ID, and fitness.
        Points have larger diameters if their fitness is higher (fitness ranges from -1 to 1).
        The plot is set to dark mode with no background walls, axes, or axes labels.
        
        Parameters:
        - session: SQLAlchemy session object.
        - population_id: The unique identifier (UUID) of the population.
        """

        # Fetch all frameworks associated with the population
        frameworks = session.query(Framework).filter_by(population_id=population_id).all()

        # Lists to store embeddings and other data
        embeddings = []
        cluster_ids = []
        framework_names = []
        framework_ids = []
        fitness_values = []

        for fw in frameworks:
            # Assuming framework_descriptor is a list of floats
            embedding = fw.framework_descriptor
            cluster_id = fw.cluster_id
            fitness = fw.framework_fitness

            if embedding and cluster_id and fitness is not None:
                embeddings.append(embedding)
                cluster_ids.append(cluster_id)
                framework_names.append(fw.framework_name)
                framework_ids.append(fw.framework_id)
                fitness_values.append(fitness)

        # Convert lists to numpy arrays
        embeddings = np.array(embeddings)
        cluster_ids = np.array(cluster_ids)
        fitness_values = np.array(fitness_values)

        # Dimensionality reduction using UMAP
        if embeddings.shape[1] >= 3:
            umap_model = UMAP(n_components=3, random_state=42)
            embeddings_3d = umap_model.fit_transform(embeddings)
        else:
            raise ValueError("Embeddings must have at least 3 dimensions for 3D plotting.")

        # Map cluster IDs to integers for coloring
        unique_clusters = list(set(cluster_ids))
        cluster_to_int = {cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)}
        cluster_labels = np.array([cluster_to_int[cid] for cid in cluster_ids])

        # Normalize fitness values to [0, 1]
        normalized_fitness = (fitness_values + 1) / 2  # Fitness ranges from -1 to 1

        # Apply quadratic scaling to amplify size differences
        scaled_fitness = normalized_fitness ** 2  # Square the normalized fitness

        # Define marker size range
        min_size = 5
        max_size = 100  # Adjusted for noticeable size difference

        # Scale to the marker size range
        sizes = min_size + scaled_fitness * (max_size - min_size)

        # Prepare DataFrame for Plotly with desired column names
        df = pd.DataFrame({
            'UMAP1': embeddings_3d[:, 0],
            'UMAP2': embeddings_3d[:, 1],
            'UMAP3': embeddings_3d[:, 2],
            'Cluster': cluster_ids,
            'Cluster_Label': cluster_labels,
            'Framework Name': framework_names,
            'Framework ID': framework_ids,
            'Fitness': fitness_values,
            'Size': sizes
        })

        # Create interactive 3D scatter plot with Plotly
        fig = px.scatter_3d(
            df,
            x='UMAP1',
            y='UMAP2',
            z='UMAP3',
            color='Cluster_Label',
            size='Size',
            hover_data=['Framework Name', 'Framework ID', 'Cluster', 'Fitness'],
            color_continuous_scale='Rainbow',
            title=f'3D UMAP Cluster Plot for Population {population_id}',
            labels={'color': 'Cluster'}
        )

        # Update the layout for dark mode and remove axes, labels, and background walls
        fig.update_layout(
            template='plotly_dark',  # Set dark mode theme
            scene=dict(
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False),
                bgcolor='rgba(0,0,0,0)'  # Make the background transparent
            ),
            legend_title_text='Cluster',
            title_font_color='white'  # Ensure the title is visible on dark background
        )

        # Update marker opacity
        fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0)))

        # Show the plot
        fig.show()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="/home/j/Documents/AgentBreeder/data/mmlu_sample_3.csv")
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='/home/j/Documents/AgentBreeder/results')
    parser.add_argument('--dataset_name', type=str, default="mmlu")
    parser.add_argument('--n_generation', type=int, default=100)
    parser.add_argument('--n_mutations', type=int, default=20)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('-mp', '--num_mutation_prompts', default=2)
    parser.add_argument('-ts', '--num_thinking_styles', default=4)
    parser.add_argument('-e', '--num_evals', default=10)
    parser.add_argument('-n', '--simulations', default=10)

    args = parser.parse_args()

    population_id = "dd615513-d3a0-43fd-bee3-31d3e87b7cc4"
   

    random.seed(args.shuffle_seed)
    
    session, Base = initialize_session()

    visualizer = Visualizer()

    visualizer.plot(session, population_id)
    
    session.close()

  
        
