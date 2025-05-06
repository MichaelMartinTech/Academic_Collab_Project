import networkx as nx
import matplotlib.pyplot as plt
import time
import os

def draw_collab_graph(G, max_nodes=100, save_path="top_authors_graph.png"):
    start_time = time.time()
    print(f"Visualizing top {max_nodes} nodes by degree...")

    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
    subgraph = G.subgraph([n for n, _ in top_nodes])
    pos = nx.spring_layout(subgraph, seed=42)

    plt.figure(figsize=(12, 12))
    nx.draw(subgraph, pos, node_size=40, alpha=0.6, edge_color='gray')
    plt.title(f"Top {max_nodes} authors (by degree)")
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()

    # For printing time
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"Graph saved to {save_path} in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")