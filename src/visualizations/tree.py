import sys

sys.path.append("src")
sys.path.append("")
import datetime
import uuid
import random
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque
from base import initialize_session, System, Population
import matplotlib.colors as mcolors
from dotenv import load_dotenv

load_dotenv(override=True)


def plot_tree(systems):
    """
    Plot a multi-layer graph from a list of System objects.
    Each unique 'generation_timestamp' becomes its own layer.
    Each 'cluster_id' is represented by a different color.
    Node labels are shown as 'system_name' rather than 'system_id'.
    Node sizes are proportional to system_capability_ci_median (0 to 1).
    If a node has system_capability_ci_median = -1 or None, treat it as 0.

    Labels for each layer are alternately drawn above and below nodes
    to reduce overlap of long names.
    """

    # 1. Collect and sort the distinct generation timestamps
    unique_gen_timestamps = sorted({s.generation_timestamp for s in systems})

    # 2. Map each timestamp to an integer "layer" index
    timestamp_to_layer = {
        gen_ts: idx for idx, gen_ts in enumerate(unique_gen_timestamps)
    }

    # 3. Initialize the graph
    G = nx.Graph()

    # 4. Add nodes with layer, cluster_id, and fitness attributes
    for sys in systems:
        # Handle fitness edge cases
        if (
            sys.system_capability_ci_median is None
            or sys.system_capability_ci_median == -1
        ):
            fitness_val = 0.0
        else:
            fitness_val = sys.system_capability_ci_median  # assumed in [0, 1]

        G.add_node(
            sys.system_id,  # node identifier = system_id
            layer=timestamp_to_layer[sys.generation_timestamp],
            cluster_id=sys.cluster_id,  # for color grouping
            fitness=fitness_val,  # store normalized fitness
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

    # 7. Assign colors by cluster_id

    all_clusters = list({G.nodes[n]["cluster_id"] for n in G.nodes()})

    cmap = plt.cm.get_cmap("rainbow", len(all_clusters))
    node_color = []
    for n in G.nodes():
        cluster = G.nodes[n]["cluster_id"]
        cluster_index = all_clusters.index(cluster)
        node_color.append(cmap(cluster_index))

    # 8. Create a label dictionary mapping system_id -> system_name
    label_dict = {sys.system_id: sys.system_name for sys in systems}

    # 9. Create a node_size list based on fitness
    #    e.g. map fitness=0 to size=100, fitness=1 to size=2100
    node_size = []
    for n in G.nodes():
        f = G.nodes[n]["fitness"]  # 0.0 <= f <= 1.0
        size = 100 + 2000 * f  # scale the node size
        node_size.append(size)

    # 10. Draw the graph WITHOUT labels
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        node_color=node_color,
        edge_color="#ddddd4",
        with_labels=False,  # don't draw labels here
        font_size=8,
        node_size=node_size,
    )

    # 11. Prepare a custom position dictionary for labels
    #     that shifts them above/below the node per layer
    label_pos = {}
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        layer_idx = data["layer"]
        # If layer is even, move label above (+), else below (-)
        offset = 0.07 if (layer_idx % 2 == 0) else -0.07
        # Create a new position with y-offset
        label_pos[node] = (x, y + offset)

    # 12. Draw labels separately, at the offset positions
    nx.draw_networkx_labels(
        G,
        label_pos,
        labels=label_dict,
        font_size=8,
    )

    plt.title(
        "Systems by Generation Timestamp (colored by cluster_id, sized by fitness)"
    )
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    population_id = "d44a351c-d454-4d2c-ae74-f7e0e88b9ce8"
    population_id = "dd43526d-9a36-41c3-89bb-2f71c7738040"

    for session in initialize_session():
        population = (
            session.query(Population)
            .order_by(Population.population_timestamp.desc())
            .limit(1)
            .one()
        )
        population_id = population.population_id

        print(population_id)

        population_id = "51d46f52-7315-419e-891b-f275172eb94a"

        # Suppose you have a list of systems from your DB:
        systems = session.query(System).filter_by(population_id=population_id).all()
        print(len(systems))
        print(systems[0])
        print(population_id)
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
