import networkx as nx
import numpy as np

def build_network(adj, nodes, threshold_mode, threshold_value, threshold_percentile):
    """Create network graph from adjacency matrix."""
    if threshold_mode == 'percentile':
        threshold = np.percentile(adj, threshold_percentile)
        title_label = f'Threshold: Top {threshold_percentile}th Percentile'
    else:
        threshold = threshold_value
        title_label = f'Threshold: >{threshold*100:.1f}%'

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for src_idx, src in enumerate(nodes):
        for tgt_idx, tgt in enumerate(nodes):
            if src_idx != tgt_idx and adj[tgt_idx, src_idx] > threshold:
                G.add_edge(src, tgt, weight=adj[tgt_idx, src_idx])
    return G, title_label
