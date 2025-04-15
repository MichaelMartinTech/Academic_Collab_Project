# Ignore this, Look at the "main" notebook instead
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

def build_author_graph(papers_tsv, authors_tsv, citations_tsv):
    print("build_author_graph called...")
    # Paper --> Year
    paper_year = {}
    with open(papers_tsv, encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[1].isdigit():
                paper_year[parts[0]] = int(parts[1])

    # Paper --> Citations
    citation_count = defaultdict(int)
    with open(citations_tsv, encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                cited = parts[1]
                citation_count[cited] += 1

    # Paper --> Authors
    paper_authors = defaultdict(list)
    with open(authors_tsv, encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                paper_id, author_id = parts[0], parts[1]
                paper_authors[paper_id].append(author_id)

    # Build coauthorship graph
    print("Creating coauthorship edges...")
    G = nx.Graph()
    for paper_id, authors in tqdm(paper_authors.items(), total=len(paper_authors)):
        year = paper_year.get(paper_id)
        cites = citation_count.get(paper_id, 0)

        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                a1, a2 = authors[i], authors[j]
                if G.has_edge(a1, a2):
                    G[a1][a2]["papers"].append(paper_id)
                    G[a1][a2]["years"].append(year)
                    G[a1][a2]["cites"].append(cites)
                else:
                    G.add_edge(a1, a2, papers=[paper_id], years=[year], cites=[cites])

    print(f"Graph has {G.number_of_nodes()} authors and {G.number_of_edges()} coauthor relationships.")
    return G