import re
import numpy as np
import torch

def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def greedy_hierarchical_clustering(similarity_matrix, num_clusters=20, subsample=1):
    seq_length = similarity_matrix.shape[0]
    change_points = []
    for i in range(num_clusters - 1):
        # print(f'Time {datetime.now().strftime("%Y/%m/%d %H:%M:%S")} Step {i} / {num_clusters}')
        best_cluster_score = -float('inf')
        best_change_point = 1
        temp_change_points = change_points.copy()
        for change_point in range(1, seq_length - 1, subsample):
            if change_point not in temp_change_points:
                temp_change_points.append(change_point)
                cluster_score = get_cluster_score(similarity_matrix, temp_change_points)
                if torch.isnan(cluster_score).item():
                    raise ValueError("Cluster score is NaN")
                temp_change_points.pop()
                if cluster_score > best_cluster_score:
                    best_cluster_score = cluster_score
                    best_change_point = change_point
        change_points.append(best_change_point)

    return sorted(change_points)

def get_cluster_score(similarity_matrix, change_points):
    if isinstance(change_points, list):
        change_points = np.sort(change_points)
    else:
        change_points, _ = torch.sort(change_points)
    sum_all_clusters = 0
    for i in range(len(change_points) + 1):
        if i == 0:
            start_idx = 0
        else:
            start_idx = change_points[i-1]
        if i == len(change_points):
            end_idx = len(similarity_matrix)
        else:
            end_idx = change_points[i]

        len_cluster = end_idx - start_idx
        sum_cluster = similarity_matrix[start_idx:end_idx, start_idx:end_idx].sum()

        sum_all_clusters += sum_cluster / len_cluster
    return sum_all_clusters
