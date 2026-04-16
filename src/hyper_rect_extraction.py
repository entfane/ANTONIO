import hdbscan
import numpy as np
from hyperrectangles import calculate_hyperrectangle
from data import load_align_mat


def compute_hyperrectangles(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
) -> list[np.ndarray]:
    embeddings = embeddings.astype(np.float64)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="cosine", algorithm='generic')
    clusterer.fit(embeddings)

    labels = clusterer.labels_

    cluster_ids = sorted(set(labels) - {-1})

    if not cluster_ids:
        print("Warning: HDBSCAN found no clusters, falling back to a single cluster.")
        labels = np.zeros(len(embeddings), dtype=int)
        cluster_ids = [0]

    center_of_rect = np.stack([
        embeddings[labels == cid].mean(axis=0) for cid in cluster_ids
    ])

    # for i, point in enumerate(embeddings):
    #     if labels[i] == -1:
    #         labels[i] = cluster_ids[np.argmin(np.linalg.norm(center_of_rect - point, axis=1))]


    rectangles = []
    align_matrices = []
    for cluster_id in cluster_ids:
        points = embeddings[labels == cluster_id]
        align_mat = load_align_mat("DATSET", "MODEL", points, False)
        points = points @ align_mat
        rect   = calculate_hyperrectangle(points)
        rectangles.append(rect)
        align_matrices.append(align_mat)

    print(f"Found {len(rectangles)} rectangles")
    return rectangles, align_matrices