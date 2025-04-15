def extract_features_from_graph(G):
    X, y = [], []
    for u, v, data in G.edges(data=True):
        num_papers = len(data['papers'])
        avg_citations = sum(data['cites']) / num_papers if num_papers else 0
        avg_year = sum(data['years']) / num_papers if num_papers else 0

        # Features
        X.append([num_papers, avg_citations, avg_year])

        # Label: strong collaboration if >= 3 papers & avg_cites >= 5
        label = 1 if (num_papers >= 3 and avg_citations >= 5) else 0
        y.append(label)
    return X, y