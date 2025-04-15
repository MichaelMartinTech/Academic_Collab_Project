import matplotlib.pyplot as plt
import networkx as nx

"""
def draw_collab_graph(G):
    plt.figure(figsize=(10, 8))
    nx.draw_spring(G, node_size=20, alpha=0.6)
    plt.title("Author Collaboration Graph")
    plt.show()
"""

def draw_collab_graph(G, max_nodes=100):
    import networkx as nx
    import matplotlib.pyplot as plt

    print(f"Visualizing top {max_nodes} nodes by degree...")
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
    subgraph = G.subgraph([n for n, _ in top_nodes])
    pos = nx.spring_layout(subgraph, seed=42)

    plt.figure(figsize=(12, 12))
    nx.draw(subgraph, pos, node_size=40, alpha=0.6, edge_color='gray')
    plt.title(f"Top {max_nodes} authors (by degree)")
    plt.axis("off")
    plt.show()