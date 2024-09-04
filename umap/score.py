import numpy as np
import pandas as pd
import math
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from typing import Sequence, Optional

from numpy.typing import ArrayLike


def calculate_cluster_centrosize(data: ArrayLike, cluster_label: ArrayLike, exclude: Optional[Sequence] = None):
    """
    Generates a matrix with different clusters on dimension 0
    and cluster name, centroid coordinates & cluster size on dimension 1.
    It will be used to compute the cluster score for a given UMAP

    Parameters
    ----------
    data : ArrayLike
        UMAP data
    cluster_label : ArrayLike
        Numpy array labeling each cluster; must be same length as data
    exclude : str or Sequence
        Labels to exclude from calculating cluster sizes

    Returns
    -------
    Numpy array with cluster_name, centroid, cluster_size on each column

    """
    cluster_uniq = np.unique(cluster_label)
    if exclude is not None:
        cluster_uniq = cluster_uniq[~np.isin(cluster_uniq, exclude)]

    centroid_list = []
    clustersize_list = []
    for cl in cluster_uniq:
        ind = cluster_label == cl
        data0 = data[ind.flatten()]
        centroid = np.median(data0, axis=0)
        # square distance between each datapoint and centroid
        square_distance = (centroid - data0) ** 2
        # median of sqrt of square_distance as cluster size
        cluster_size = np.median(np.sqrt(square_distance[:, 0] + square_distance[:, 1]))
        centroid_list.append(centroid)
        clustersize_list.append(cluster_size)
    cluster_matrix = np.vstack([cluster_uniq, np.vstack(centroid_list).T, clustersize_list]).T
    # cluster_matrix = calculate_cluster_centrosize(umap_data, label_data, 'others' if '_others' in colormap else None)
    intra_cluster = np.median(cluster_matrix[:, -1].astype(float))
    inter_cluster = np.std(cluster_matrix[:, 1:-1].astype(float))
    cluster_score = inter_cluster / intra_cluster
    return cluster_score



def ARI(label,pred):
    return metrics.adjusted_rand_score(label,pred)


def NMI(label,pred):
    return metrics.normalized_mutual_info_score(label,pred)


def FMI(label,pred):
    return metrics.fowlkes_mallows_score(label,pred)


def Purity(label,pred,uniq):
    sum = 0
    pred = np.reshape(pred,(-1,1))
    for i,da in enumerate(uniq):
        ind = np.isin(label[:],da)
        p = pred[ind,:]
        a = np.sum(p == da)
        sum +=a
    return float(sum/len(label))


def SC(data,pred):
    return metrics.silhouette_score(data,pred)


def CH(data,pred):
    return metrics.calinski_harabasz_score(data,pred)


def davies_bouldin_score(X, labels):
    n_clusters = len(np.unique(labels))
    cluster_centers = np.zeros((n_clusters, X.shape[1]))

    for i in range(n_clusters):
        cluster_centers[i] = np.mean(X[labels == i], axis=0)

    pairwise_distances_centers = pairwise_distances(cluster_centers)
    cluster_distances = np.zeros(n_clusters)

    for i in range(n_clusters):
        cluster_distances[i] = np.max(pairwise_distances_centers[i, labels == i])

    davies_bouldin = 0.0
    for i in range(n_clusters):
        max_ratio = -np.inf
        for j in range(n_clusters):
            if i != j:
                ratio = (cluster_distances[i] + cluster_distances[j]) / pairwise_distances_centers[i, j]
                if ratio > max_ratio:
                    max_ratio = ratio
        davies_bouldin += max_ratio

    davies_bouldin /= n_clusters
    return davies_bouldin


def dunn_index(X, labels):
    n_clusters = len(np.unique(labels))
    cluster_distances = pairwise_distances(X)

    min_inter_cluster_distance = np.inf
    max_intra_cluster_distance = -np.inf

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            inter_cluster_distance = np.min(cluster_distances[labels == i][:, labels == j])
            min_inter_cluster_distance = min(min_inter_cluster_distance, inter_cluster_distance)

        intra_cluster_distance = np.max(cluster_distances[labels == i])
        max_intra_cluster_distance = max(max_intra_cluster_distance, intra_cluster_distance)

    dunn = min_inter_cluster_distance / max_intra_cluster_distance

    return dunn


def dis_cal(cen,all):
    sum=0
    for i in range(len(all)):
        number = math.sqrt((all[i][0]-cen[0])**2+(all[i][1]-cen[1])**2)
        sum += number**2
    return sum


def Cen_cal(uniq,label,data):
    da = {
        # 'group':uniq_group,
        'x':[],
        'y':[],
        'sum':[],}
    
    for i in uniq:
        ind = np.isin(label[:],i)
        embedding = data[ind,:]
        center = np.mean(embedding,axis=0)
        da['x'].append(center[0])
        da['y'].append(center[1])
        da['sum'].append(dis_cal(center,embedding))

    df = pd.DataFrame(da,index=uniq)
    print(df)
    sum = df['sum'].sum()

    return sum


def F1(label,pred,beta=1.):
    (tn, fp), (fn, tp) = metrics.cluster.pair_confusion_matrix(label, pred)
    ri = (tp + tn) / (tp + tn + fp + fn)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    return f_beta


