import argparse
import random
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
from bayesian_illumination import Generator, initialize_population_id, generate_mutant
from descriptor import Clusterer, Visualizer
from base import initialize_session, Population, Framework
from evaluator import Evaluator
import time  # Added for restart delay

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine
from umap import UMAP
import numpy as np

# Assuming you have already set up your SQLAlchemy engine
# engine = create_engine('your_database_url')
# session_factory = sessionmaker(bind=engine)
# Session = scoped_session(session_factory)
# session = Session()

def plot_clusters_3d(session, population_id):
    """
    Plots the clusters in 3D for a given population using UMAP for dimensionality reduction.

    Parameters:
    - session: SQLAlchemy session object.
    - population_id: The unique identifier (UUID) of the population.
    """

    # Fetch the population
    population = session.query(Population).filter_by(population_id=population_id).one()

    # Fetch all frameworks associated with the population
    frameworks = session.query(Framework).filter_by(population_id=population_id).all()

    # Lists to store embeddings and cluster IDs
    embeddings = []
    cluster_ids = []

    for fw in frameworks:
        embedding = fw.framework_descriptor  # Assuming this is a list of floats
        cluster_id = fw.cluster_id

        if embedding and cluster_id:
            embeddings.append(embedding)
            cluster_ids.append(cluster_id)

    # Convert lists to numpy arrays
    embeddings = np.array(embeddings)
    cluster_ids = np.array(cluster_ids)

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

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=cluster_labels,
        cmap='tab10',
        depthshade=True,
        s=50,
        alpha=0.8
    )

    # Add legend
    handles, _ = scatter.legend_elements()
    legend_labels = [f"Cluster {cluster_to_int[cid]}" for cid in unique_clusters]
    ax.legend(handles, legend_labels, title="Clusters")

    # Set labels
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_zlabel('UMAP Dimension 3')
    ax.set_title(f'3D UMAP Cluster Plot for Population {population_id}')

    plt.show()
import plotly.express as px
import numpy as np
import pandas as pd
from umap import UMAP

def plot_clusters_3d_interactive(session, population_id):
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



def main(args, population_id=None):
    random.seed(args.shuffle_seed)
    
    session, Base = initialize_session()

    plot_clusters_3d_interactive(session, population_id)
    
    # # Re-load the population object in this session
    # population = session.query(Population).filter_by(
    #     population_id=population_id
    # ).one()

    # # Recluster the population
    # visualizer = Visualizer()
    # visualizer.plot(population)
    session.close()

    return population_id  # Return the population ID for restarts


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

    population_id = "68d8c6e5-d514-4119-ba15-eaa12d09e3f0"
   
    population_id = main(args, population_id)
        
