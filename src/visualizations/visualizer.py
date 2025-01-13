import sys

sys.path.append("src")
import uuid
import random
import pandas as pd
import numpy as np
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy.orm import Session
from base import initialize_session, System, Population


class Visualizer:
    def plot(self, session: Session, population_id: str):
        """
        Plots the clusters in 3D for a given population using UMAP for dimensionality reduction.
        Hovering over a point displays the multi-agent system name, system ID, and fitness.
        Points have larger diameters if their fitness is higher (fitness ranges from -1 to 1).
        The plot is set to dark mode with no background walls, axes, or axes labels.

        Parameters:
        - session: SQLAlchemy session object.
        - population_id: The unique identifier (UUID) of the population.
        """

        # Fetch all systems associated with the population
        systems = session.query(System).filter_by(population_id=population_id).all()
        print("Number of systems fetched:", len(systems))

        # Lists to store embeddings and other data
        embeddings = []
        cluster_ids = []
        system_names = []
        system_ids = []
        fitness_values = []

        # Count number of unique cluster_ids
        cluster_id_set = {fw.cluster_id for fw in systems if fw.cluster_id}
        print("Number of unique clusters:", len(cluster_id_set))

        # Use a random UUID to replace None cluster IDs
        null_cluster_id = str(uuid.uuid4())

        for fw in systems:
            if fw.system_descriptor:
                # Assuming system_descriptor is a list of floats
                embedding = fw.system_descriptor
                cluster_id = fw.cluster_id if fw.cluster_id else null_cluster_id
                fitness = fw.system_fitness if fw.system_fitness is not None else -1

                if embedding and cluster_id and (fitness is not None):
                    embeddings.append(embedding)
                    cluster_ids.append(cluster_id)
                    system_names.append(fw.system_name)
                    system_ids.append(fw.system_id)
                    fitness_values.append(fitness)

        # Convert lists to numpy arrays
        embeddings = np.array(embeddings)
        cluster_ids = np.array(cluster_ids)
        fitness_values = np.array(fitness_values)

        # Dimensionality reduction using UMAP
        if embeddings.shape[1] < 3:
            raise ValueError(
                "Embeddings must have at least 3 dimensions for 3D plotting."
            )

        umap_model = UMAP(n_components=3, random_state=42)
        embeddings_3d = umap_model.fit_transform(embeddings)

        # Map cluster IDs to integers for coloring
        unique_clusters = list(set(cluster_ids))
        cluster_to_int = {
            cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)
        }
        cluster_labels = np.array([cluster_to_int[cid] for cid in cluster_ids])

        # Normalize fitness values to [0, 1]
        # Fitness is in [-1, 1], so add 1 and divide by 2
        normalized_fitness = (fitness_values + 1) / 2
        # Apply quadratic scaling to amplify size differences
        scaled_fitness = normalized_fitness**2

        # Define marker size range
        min_size = 5
        max_size = 100  # Adjusted for more noticeable differences

        # Scale to the marker size range
        sizes = min_size + scaled_fitness * (max_size - min_size)

        # Prepare DataFrame for Plotly with desired column names
        df = pd.DataFrame(
            {
                "UMAP1": embeddings_3d[:, 0],
                "UMAP2": embeddings_3d[:, 1],
                "UMAP3": embeddings_3d[:, 2],
                "Cluster": cluster_ids,
                "Cluster_Label": cluster_labels,
                "System Name": system_names,
                "System ID": system_ids,
                "Fitness": fitness_values,
                "Size": sizes,
            }
        )

        # Create the 3D scatter plot
        fig = px.scatter_3d(
            df,
            x="UMAP1",
            y="UMAP2",
            z="UMAP3",
            color="Cluster_Label",
            size="Size",
            hover_data=["System Name", "System ID", "Cluster", "Fitness"],
            color_continuous_scale="Rainbow",
            title=f"3D UMAP Cluster Plot for Population {population_id}",
            labels={"color": "Cluster"},
        )

        # Update the layout for dark mode, remove axes, labels, and background walls
        fig.update_layout(
            template="plotly_dark",
            scene=dict(
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False),
                bgcolor="rgba(0,0,0,0)",
            ),
            legend_title_text="Cluster",
            title_font_color="white",
        )

        # Update marker opacity
        fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0)))

        # --------------------------------------------------------------------
        #  Add lines (edges) from child to parent(s), colored by parent's cluster
        # --------------------------------------------------------------------
        # 1) Create dictionaries for quick lookup: system_id -> (x, y, z) coords
        #    and system_id -> cluster_label
        system_id_to_coords = {}
        system_id_to_cluster_label = {}

        for i in range(len(df)):
            sid = df.loc[i, "System ID"]
            system_id_to_coords[sid] = (
                df.loc[i, "UMAP1"],
                df.loc[i, "UMAP2"],
                df.loc[i, "UMAP3"],
            )
            system_id_to_cluster_label[sid] = df.loc[i, "Cluster_Label"]

        # 2) We'll use the same "Rainbow" color scale that the scatter uses.
        #    For convenience, we can sample from plotly's built-in scales.
        #    The index for the color scale is parent's cluster_label normalized
        #    by the max cluster label (since cluster labels go from 0..N-1).
        #    Plotly provides a function to sample a continuous color scale:
        from plotly.colors import sample_colorscale

        max_label = cluster_labels.max() if len(cluster_labels) > 0 else 1

        # Helper function to map a cluster_label to an RGBA color in "Rainbow" scale
        def get_line_color(parent_label):
            if max_label == 0:
                # Avoid division by zero if only one cluster
                return px.colors.sample_colorscale("Rainbow", 0.5)[0]
            frac = parent_label / float(max_label)
            return sample_colorscale("Rainbow", frac)[0]  # returns an RGBA string

        # 3) Add a 3D line trace for each (child -> parent) relationship
        #    Each system can have up to two parents (system_first_parent_id, system_second_parent_id).
        for system in systems:
            child_id = system.system_id
            child_coords = system_id_to_coords.get(child_id)

            if child_coords:
                # Handle first parent
                if system.system_first_parent_id is not None:
                    parent_id = system.system_first_parent_id
                    parent_coords = system_id_to_coords.get(parent_id)
                    if parent_coords:
                        parent_label = system_id_to_cluster_label.get(parent_id, 0)
                        color_line = get_line_color(parent_label)

                        fig.add_trace(
                            go.Scatter3d(
                                x=[child_coords[0], parent_coords[0]],
                                y=[child_coords[1], parent_coords[1]],
                                z=[child_coords[2], parent_coords[2]],
                                mode="lines",
                                line=dict(color=color_line, width=2),
                                hoverinfo="none",
                                showlegend=False,
                            )
                        )

                # Handle second parent
                if system.system_second_parent_id is not None:
                    parent_id = system.system_second_parent_id
                    parent_coords = system_id_to_coords.get(parent_id)
                    if parent_coords:
                        parent_label = system_id_to_cluster_label.get(parent_id, 0)
                        color_line = get_line_color(parent_label)

                        fig.add_trace(
                            go.Scatter3d(
                                x=[child_coords[0], parent_coords[0]],
                                y=[child_coords[1], parent_coords[1]],
                                z=[child_coords[2], parent_coords[2]],
                                mode="lines",
                                line=dict(color=color_line, width=2),
                                hoverinfo="none",
                                showlegend=False,
                            )
                        )

        # Finally, show the plot
        fig.show()


if __name__ == "__main__":
    # Example usage:

    random.seed(42)

    for session in initialize_session():
        population = (
            session.query(Population)
            .order_by(Population.population_timestamp.desc())
            .limit(1)
            .one()
        )
        print("population_id:", population.population_id)

        visualizer = Visualizer()
        visualizer.plot(session, population.population_id)

        session.close()
