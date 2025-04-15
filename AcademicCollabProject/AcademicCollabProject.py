from src.preprocessing import build_author_graph
from src.features import extract_features_from_graph
from src.model import train_model, evaluate_model, explain_model
from src.visualize import draw_collab_graph

import os

def main():
    # Point to file paths
    raw_data_dir = os.path.join("data", "raw")

    papers_file = os.path.join(raw_data_dir, "Papers_CS_20190919.tsv")
    authors_file = os.path.join(raw_data_dir, "PAuAf_CS_20190919.tsv")
    citations_file = os.path.join(raw_data_dir, "PR_CS_20190919.tsv")

    print("Building author collaboration graph...")
    G = build_author_graph(papers_file, authors_file, citations_file)

    print("Visualizing coauthor network...")
    draw_collab_graph(G)

    print("Extracting features for ML...")
    X, y = extract_features_from_graph(G)
    print(f"{len(X)} collaboration pairs extracted.")

    print("Training model...")
    model = train_model(X, y)
    evaluate_model(model, X, y)

    print("Interpreting model...")
    explain_model(model, X)

if __name__ == "__main__":
    main()
