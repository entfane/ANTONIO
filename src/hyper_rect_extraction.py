import hdbscan
import numpy as np
from hyperrectangles import calculate_hyperrectangle


def compute_hyperrectangles(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
) -> list[np.ndarray]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="cosine", algorithm='generic')
    clusterer.fit(embeddings)

    labels = clusterer.labels_

    cluster_ids = sorted(set(labels) - {-1})

    center_of_rect = np.stack([
        embeddings[labels == cid].mean(axis=0) for cid in cluster_ids
    ])

    for i, point in enumerate(embeddings):
        if labels[i] == -1:
            labels[i] = cluster_ids[np.argmin(np.linalg.norm(center_of_rect - point, axis=1))]


    rectangles = []
    for cluster_id in cluster_ids:
        points = embeddings[labels == cluster_id]
        rect   = calculate_hyperrectangle(points)
        rectangles.append(rect)

    print(f"Found {len(rectangles)} rectangles")
    return rectangles