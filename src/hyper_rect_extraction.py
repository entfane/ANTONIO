import hdbscan
import numpy as np
from hyperrectangles import calculate_hyperrectangle


def compute_hyperrectangles(
    embeddings: np.ndarray,
    min_cluster_size: int = 50,
    metric: str = 'cosine',
) -> list[np.ndarray]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    clusterer.fit(embeddings)

    labels = clusterer.labels_

    rectangles = []
    for cluster_id in sorted(set(labels) - {-1}):
        points = embeddings[labels == cluster_id]
        rect   = calculate_hyperrectangle(points)
        rectangles.append(rect)

    print(f"Found {len(rectangles)} rectangles")
    return rectangles