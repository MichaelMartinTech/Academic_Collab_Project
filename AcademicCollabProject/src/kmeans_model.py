{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "330a2199-6f6a-4221-bb77-8e5118ffd197",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics import silhouette_score\n",
    "import pandas as pd\n",
    "import pickle\n",
    "\n",
    "def run_kmeans(X, n_clusters=2, save_path=\"kmeans_clustered_output.csv\"):\n",
    "    kmeans = KMeans(n_clusters=n_clusters, random_state=42)\n",
    "    cluster_labels = kmeans.fit_predict(X)\n",
    "\n",
    "    silhouette = silhouette_score(X, cluster_labels)\n",
    "\n",
    "    # Save labeled data\n",
    "    clustered_df = pd.DataFrame(X, columns=[\"#papers\", \"avg_cites\", \"avg_year\"])\n",
    "    clustered_df[\"cluster\"] = cluster_labels\n",
    "    clustered_df.to_csv(save_path, index=False)\n",
    "    print(f\"Saved KMeans results to {save_path}\")\n",
    "    print(f\"Silhouette Score: {silhouette:.3f}\")\n",
    "\n",
    "    # Save model if needed\n",
    "    with open(\"kmeans_model.pkl\", \"wb\") as f:\n",
    "        pickle.dump(kmeans, f)\n",
    "\n",
    "    return kmeans, cluster_labels"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
