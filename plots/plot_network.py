import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

def plot_network_graph(G, fevd_horizon, title_label):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.axis('off')
    if len(G) == 0:
        return fig

    node_color = "#1f77b4"
    node_alpha = 0.7
    node_size = 600
    node_radius = (node_size ** 0.5)/50
    nodes_list = list(G.nodes())
    n = len(nodes_list)
    radius = 4
    angle_step = 2 * 3.1416 / n
    pos = {}
    for i, node in enumerate(nodes_list):
        angle = i * angle_step
        pos[node] = (radius * np.cos(angle), radius * np.sin(angle))

    def adjust_line(x1,y1,x2,y2,r):
        dx, dy = x2-x1, y2-y1
        dist = (dx**2 + dy**2)**0.5
        if dist == 0: return (x1,y1),(x2,y2)
        x1_new, y1_new = x1+dx/dist*r, y1+dy/dist*r
        x2_new, y2_new = x2-dx/dist*r, y2-dy/dist*r
        return (x1_new, y1_new),(x2_new, y2_new)

    for src,tgt in G.edges():
        (x1,y1),(x2,y2) = adjust_line(*pos[src],*pos[tgt],node_radius)
        arrow = FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',color='lightgray',linewidth=1.5,alpha=0.6,mutation_scale=12)
        ax.add_patch(arrow)

    for node,(x,y) in pos.items():
        circle = Circle((x,y),node_radius,color=node_color,alpha=node_alpha,ec='black',lw=1.5)
        ax.add_patch(circle)
        ax.text(x,y,node[:2].upper(),ha='center',va='center',fontsize=11,fontweight='bold')

    margin = 1.5
    ax.set_xlim(-radius-margin,radius+margin)
    ax.set_ylim(-radius-margin,radius+margin)
    ax.set_aspect('equal','box')
    ax.set_title(f'Contagion Network (FEVD={fevd_horizon} days)\n{title_label}',fontsize=10,fontweight='bold',pad=10)
    st.pyplot(fig)
