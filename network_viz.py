import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, Circle
import streamlit as st

def build_network(nodes: list, adj: np.ndarray, fixed_threshold: float = None, percentile: int = None):
    """Build directed graph based on FEVD adjacency matrix."""
    if percentile is not None:
        threshold = np.percentile(adj, percentile)
        title_label = f'Threshold: Top {percentile}th Percentile'
    else:
        threshold = fixed_threshold or 0.02
        title_label = f'Threshold: >{threshold*100:.1f}%'
    
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    
    for src_idx, src in enumerate(nodes):
        for tgt_idx, tgt in enumerate(nodes):
            if src_idx != tgt_idx and adj[tgt_idx, src_idx] > threshold:
                G.add_edge(src, tgt, weight=adj[tgt_idx, src_idx])
    
    return G, title_label

@st.cache_data(show_spinner="Drawing network...")
def draw_enhanced_network(_G, fevd_horizon: int, threshold_label: str):
    """Draw the contagion network visualization."""
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    ax.axis('off')
    
    if len(_G) == 0:
        return fig
    
    node_color, node_alpha = "#1f77b4", 0.8
    node_radius = 0.08
    
    nodes_list = list(_G.nodes())
    n = len(nodes_list)
    radius = 0.6
    pos = {node: (radius * np.cos(i * 2 * np.pi / n), radius * np.sin(i * 2 * np.pi / n)) 
           for i, node in enumerate(nodes_list)}
    
    def adjust_line(x1, y1, x2, y2, r):
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        if dist == 0:
            return (x1, y1), (x2, y2)
        return (x1 + dx/dist*r, y1 + dy/dist*r), (x2 - dx/dist*r, y2 - dy/dist*r)
    
    for src, tgt in _G.edges():
        (x1, y1), (x2, y2) = adjust_line(*pos[src], *pos[tgt], node_radius)
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', color='#666666',
                                linewidth=0.4, alpha=0.8, mutation_scale=4, zorder=1,
                                connectionstyle="arc3,rad=0.08")
        ax.add_patch(arrow)
    
    for node, (x, y) in pos.items():
        circle = Circle((x, y), radius=node_radius, color=node_color, alpha=node_alpha,
                        ec='#222222', lw=0.3, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, node[:2].upper(), ha='center', va='center', fontsize=4, fontweight='600', color='white', zorder=10)
    
    margin = 0.2
    ax.set_xlim(-radius-margin, radius+margin)
    ax.set_ylim(-radius-margin, radius+margin)
    ax.set_aspect('equal', 'box')
    ax.set_title(f'Contagion Network (FEVD={fevd_horizon} days)\n{threshold_label}', fontsize=6, fontweight='600', pad=4)
    plt.tight_layout(pad=0.1)
    
    return fig
