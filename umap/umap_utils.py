import umap
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from matplotlib import cm

from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from scipy.stats import pearsonr
from tqdm import tqdm

import os

def umap2d(args,embed,labels,title,xlabel,ylabel,s,alpha,show_legend): #,group_annotation):
    reducer = umap.UMAP(n_neighbors=50,n_components=2, min_dist=0.1, metric='euclidean', verbose=True)
    reducer.fit(embed.reshape(embed.shape[0], -1))
    umap_data = reducer.transform(embed.reshape(embed.shape[0], -1))

    colormap = 'tab20_others'

    #label  data
    unique_groups = np.unique(labels)
    # unique_groups = np.unique(group_annotation[:,1])
    label_converted = labels[:].astype(object)
    # label_converted[:] = 'others'
    for gp in unique_groups:
        ind = np.where(label_converted == gp)
        label_converted[ind] = gp

    #plot
    savepath = join(args.savepath_dict["umap_figures"], title+'.png')
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap.replace('_others', '')).colors
    else:
        cmap = colormap
    fig,ax = plt.subplots(1,figsize=(7,7))
    i = 0
    for gp in unique_groups: # for gp in unique_groups:
        if '_others' in colormap and gp == 'others':
            _c = cm.Greys(25)
        else:
            _c = cmap[i % len(cmap)]
            i += 1
        ind = np.where(label_converted == gp)
        ax.scatter(
            umap_data[ind, 0],
            umap_data[ind, 1],
            s=s,
            alpha=alpha,
            c=np.array(_c).reshape(1, -1),
            label=gp,
            zorder=0 #if gp == 'others' else len(unique_groups) - i + 1,
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    hndls, names = ax.get_legend_handles_labels()
    leg = ax.legend(
        hndls,
        names,
        prop={'size': 6},
        bbox_to_anchor=(1, 1),
        loc='upper left',
        ncol=1 + len(names) // 20,
        frameon=False,
    )
    for ll in leg.legendHandles:
        ll._sizes = [6]
        ll.set_alpha(1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    savepath
    if savepath:
        fig.savefig(savepath, dpi=300)
        print(f'Figure saved to {savepath}')
    return umap_data

def umap_init_savepath(args):
    args.savepath_dict = {}
    args.savepath_dict["umap_path"]=join(args.homepath, 'analysis')
    folders = ['umap_figures', 'umap_data', 'feature_spectra_figures', 'feature_spectra_data']
    for f in folders:
        p = join(args.savepath_dict["umap_path"], f)
        if not os.path.exists(p):
            os.makedirs(p)
        args.savepath_dict[f] = p

def corr_single(offset: int, array: ArrayLike, matrix: ArrayLike, dims: int) -> ArrayLike:
    """
    Compute pearson's correlation between an array & a matrix

    Parameters
    ----------
    offset : int
        Offset
    array : ArrayLike
        1D Numpy array
    matrix : ArrayLike
        Numpy array
    dims : int
        Shape size in the second dimension of the correlation

    Returns
    -------
    Numpy array

    """
    corr = np.zeros((1, dims))
    for ii, row in enumerate(matrix):
        corr[:, ii + offset] = pearsonr(array, row)[0]
    return corr

def selfpearson_multi(data: ArrayLike, num_workers: int = 1) -> ArrayLike:
    """
    Compute self pearson correlation using multiprocessing

    Parameters
    ----------
    data : ArrayLike
        2D Numpy array; self-correlation is performed on axis 1 and iterates over axis 0
    num_workers : int
        Number of workers

    Returns
    -------
    2D Numpy array

    """
    corr = Parallel(n_jobs=num_workers, prefer='threads')(
        delayed(corr_single)(i, row, data[i:], data.shape[0]) for i, row in enumerate(tqdm(data))
    )
    corr = np.vstack(corr)
    corr_up = np.triu(corr, k=1)
    return corr_up.T + corr

