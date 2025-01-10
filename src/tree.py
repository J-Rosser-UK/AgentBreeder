import datetime
import uuid
import random
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque
from base import initialize_session, System
import matplotlib.colors as mcolors


def plot_tree(systems):
    """
    Plot a multi-layer graph from a list of System objects.
    Each unique 'generation_timestamp' becomes its own layer.
    Each 'cluster_id' is represented by a different color.
    Node labels are shown as 'system_name' rather than 'system_id'.
    """

    # 1. Collect and sort the distinct generation timestamps
    unique_gen_timestamps = sorted({s.generation_timestamp for s in systems})

    # 2. Map each timestamp to an integer "layer" index
    timestamp_to_layer = {
        gen_ts: idx for idx, gen_ts in enumerate(unique_gen_timestamps)
    }

    # 3. Initialize the graph
    G = nx.Graph()

    # 4. Add nodes with layer and cluster_id attributes
    for sys in systems:
        G.add_node(
            sys.system_id,  # node identifier stays as the UUID
            layer=timestamp_to_layer[sys.generation_timestamp],
            cluster_id=sys.cluster_id,  # Store cluster_id for coloring
        )

    # 5. Add edges from parents to child (if they exist in this population)
    system_ids = set(s.system_id for s in systems)
    for sys in systems:
        if sys.system_first_parent_id and sys.system_first_parent_id in system_ids:
            G.add_edge(sys.system_first_parent_id, sys.system_id)
        if sys.system_second_parent_id and sys.system_second_parent_id in system_ids:
            G.add_edge(sys.system_second_parent_id, sys.system_id)

    # 6. Use the built-in multipartite layout keyed by 'layer'
    pos = nx.multipartite_layout(G, subset_key="layer")

    # 7. Assign colors: all nodes sharing the same cluster_id get the same color
    all_clusters = sorted({G.nodes[n]["cluster_id"] for n in G.nodes()})
    cmap = plt.cm.get_cmap("rainbow", len(all_clusters))
    node_color = []
    for n in G.nodes():
        cluster = G.nodes[n]["cluster_id"]
        cluster_index = all_clusters.index(cluster)
        node_color.append(cmap(cluster_index))

    # 8. Create a label dictionary mapping system_id -> system_name
    label_dict = {sys.system_id: sys.system_name for sys in systems}

    # 9. Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        node_color=node_color,
        labels=label_dict,  # <-- label each node by system_name
        edge_color="grey",
        with_labels=True,
        font_size=8,
    )
    plt.title("Systems by Generation Timestamp (colored by cluster_id)")
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    population_id = "0bd59045-aa13-49ed-85f0-020a47a931f1"

    session, Base = initialize_session()
    # Example: Suppose you have a list of systems from your DB:
    systems = session.query(System).filter_by(population_id=population_id).all()

    plot_tree(systems)


# def assign_generations(systems):
#     base_generation = systems[:7]
#     generation_timestamp = datetime.datetime.utcnow()
#     for system in base_generation:
#         print(system.system_name, system.system_id)
#         system.update(generation_timestamp=generation_timestamp)

#     other_generations = systems[7:]

#     # each generation has 10 systems
#     for i in range(5):
#         generation = other_generations[i * 10 : (i + 1) * 10]
#         generation_timestamp = datetime.datetime.utcnow()
#         for system in generation:
#             print(system.system_name, system.system_id)
#             system.update(generation_timestamp=generation_timestamp)
