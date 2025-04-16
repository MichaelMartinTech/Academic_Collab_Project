import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

print("=== Step 1: Loading the TSV files ===")
papers_df = pd.read_csv("AcademicCollabProject/data/Papers_CS_20190919.tsv", sep="\t")
papers_df.dropna(inplace=True)
papers_df.reset_index(drop=True, inplace=True)

authors_df = pd.read_csv("AcademicCollabProject/data/PAuAf_CS_20190919.tsv", sep="\t")
authors_df.dropna(inplace=True)
authors_df.reset_index(drop=True, inplace=True)

citations_df = pd.read_csv("AcademicCollabProject/data/PR_CS_20190919.tsv", sep="\t")
citations_df.dropna(inplace=True)
citations_df.reset_index(drop=True, inplace=True)

print("Loaded datasets.")
print(papers_df.head())
print(authors_df.head())
print(citations_df.head())

print("\n=== Step 2: Preprocessing citation counts ===")
citation_count = defaultdict(int)
for ref_id in citations_df["ReferenceId"]:
    citation_count[ref_id] += 1

print("Building paper metadata.")
paper_meta = {}
for _, row in papers_df.iterrows():
    paper_id = row["PaperId"]
    year = row["PublishYear"]
    estimated_citation = row["EstimatedCitation"]
    paper_meta[paper_id] = (year, estimated_citation, citation_count.get(paper_id, 0))

print("Metadata complete.")

print("\n=== Step 3: Building author features ===")
author_stats = defaultdict(lambda: {
    "paper_count": 0,
    "total_est_citations": 0,
    "total_true_citations": 0,
    "coauthors": set(),
    "active_years": set()
})

print("Grouping paper authors.")
paper_authors = defaultdict(list)
for _, row in authors_df.iterrows():
    paper_id = row["PaperSeqid"]
    author_id = row["AuthorSeqid"]
    paper_authors[paper_id].append(author_id)

print("Calculating stats per author.")
for paper_id, authors in tqdm(paper_authors.items(), desc="Processing authors"):
    meta = paper_meta.get(paper_id)
    if not meta:
        continue
    year, est_cite, true_cite = meta
    for author in authors:
        author_stats[author]["paper_count"] += 1
        author_stats[author]["total_est_citations"] += est_cite
        author_stats[author]["total_true_citations"] += true_cite
        author_stats[author]["coauthors"].update(set(authors) - {author})
        author_stats[author]["active_years"].add(year)

print("Converting author stats to DataFrame.")
data = []
for author, stats in author_stats.items():
    data.append({
        "author_id": author,
        "paper_count": stats["paper_count"],
        "avg_est_citations": stats["total_est_citations"] / stats["paper_count"],
        "avg_true_citations": stats["total_true_citations"] / stats["paper_count"],
        "unique_coauthors": len(stats["coauthors"]),
        "active_years": len(stats["active_years"])
    })

df = pd.DataFrame(data)
print("Author DataFrame created.")

print("\n=== Step 4: Normalizing features ===")
features = ["paper_count", "avg_est_citations", "avg_true_citations", "unique_coauthors", "active_years"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features normalized.")

print("\n=== Step 5: Elbow Method ===")
inertias = []
K_range = range(1, 10)

for k in K_range:
    print(f"Fitting KMeans for k = {k}...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel("k (number of clusters)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# print("\n=== Step 6: Silhouette Scores ===")
# for k in range(2, 10):
#     print(f"Calculating silhouette score for k = {k}...")
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X_scaled)
#     score = silhouette_score(X_scaled, labels)
#     print(f"k = {k}: score = {score:.4f}")

print("\n=== Step 6: Final Clustering ===")
k = 7 
print(f"Running KMeans with k = {k}...")
final_kmeans = KMeans(n_clusters=k, random_state=42)
df["cluster"] = final_kmeans.fit_predict(X_scaled)

print("\nSample authors with cluster labels:")
print(df[["author_id", "cluster"]].head())

print("Saving clustered data to CSV...")
df.to_csv("clustered_authors.csv", index=False)

print("\n=== Step 7: Visualizing Clusters with PCA ===")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["cluster"], cmap='tab10', alpha=0.6)
plt.title("K-Means Clusters (PCA 2D Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.show()

print("All steps complete.")
