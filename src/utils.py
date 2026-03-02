#utils.py
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from numpy.linalg import eig
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from scipy.stats import zscore
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from statsmodels.miscmodels.ordinal_model import OrderedModel
import plotly.graph_objects as go
from IPython.display import HTML
import scanpy as sc
import scipy as sp
from scipy.stats import fisher_exact, kendalltau
import seaborn as sb
from scipy.sparse import csr_matrix
from scipy.io import mmread
import qnorm
import torch
import torch.nn as nn
import torch.optim as optim
import qnorm
from scipy import sparse
import numba
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
import matplotlib.gridspec as gridspec
from anndata import AnnData
from scipy.sparse import issparse
import importlib.resources
from pathlib import Path
from typing import List, Tuple, Optional, Literal
from scipy.stats import mannwhitneyu
import anndata
from logging import getLogger
import time
from tqdm import tqdm  #
from itertools import product
import networkx as nx
from adjustText import adjust_text
from esda.moran import Moran_BV
import squidpy as sq
import libpysal
from matplotlib.patches import Patch
import gc
from PyComplexHeatmap import DotClustermapPlotter,HeatmapAnnotation,anno_simple,anno_label,AnnotationBase
logger = getLogger(__name__)


def counts2FPKM(counts, genelen):
    genelen = pd.read_csv(genelen, sep=',')
    genelen['TranscriptLength'] = genelen['Transcript end (bp)'] - genelen['Transcript start (bp)']
    genelen = genelen[['Gene name', 'TranscriptLength']]
    genelen = genelen.groupby('Gene name').max()

    inter = counts.columns.intersection(genelen.index)
    if len(inter) == 0:
        raise ValueError("No overlapping genes found between counts and gene length data.")

    samplename = counts.index
    counts = counts[inter].values
    genelen = genelen.loc[inter].T.values

    totalreads = counts.sum(axis=1)
    fpkm = counts * 1e9 / (genelen * totalreads.reshape(-1, 1))
    fpkm_df = pd.DataFrame(fpkm, columns=inter, index=samplename)
    
    return fpkm_df
    
def FPKM2TPM(fpkm):
    genename = fpkm.columns
    samplename = fpkm.index
    fpkm = fpkm.values
    total = fpkm.sum(axis=1).reshape(-1, 1)
    fpkm = fpkm * 1e6 / total
    tpm = pd.DataFrame(fpkm, columns=genename, index=samplename)
    return tpm



def counts2TPM(counts, genelen):
    fpkm = counts2FPKM(counts, genelen)
    tpm = FPKM2TPM(fpkm)
    return tpm

def counts2log1tpm_raw(adata, genelen_file=None):
    """
    Convert the count matrix of adata to TPM format (in sparse matrix mode) and perform log-normalization.

    Parameters:
    adata (AnnData): AnnData object.
    genelen_file (str): Path to the gene length file. If None, use the default file in the package.

    Returns:
    adata (AnnData): Converted AnnData object, with adata.X as a sparse matrix.
    """
    if genelen_file is None:
        with importlib.resources.path("PopCILA.data", "GeneLength.txt") as default_path:
            genelen_file = default_path

    if isinstance(genelen_file, Path):
        genelen_file = str(genelen_file)

    adata = adata[:, ~adata.var_names.duplicated()].copy()

    counts = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)

    tpm = counts2TPM(counts, genelen_file)

    genes_in_tpm = tpm.columns

    adata = adata[:, adata.var_names.isin(genes_in_tpm)].copy()

    tpm = tpm.loc[:, adata.var_names]

    tpm_sparse = csr_matrix(tpm.values)

    adata.X = tpm_sparse
    adata.var_names = tpm.columns
    adata.obs_names = tpm.index

    sc.pp.log1p(adata)

    return adata


def _load_gene_length(genelen_file):
    gl = pd.read_csv(genelen_file, sep=",")
    gl["TranscriptLength"] = gl["Transcript end (bp)"] - gl["Transcript start (bp)"]
    gl = gl[["Gene name", "TranscriptLength"]]
    gl = gl.groupby("Gene name")["TranscriptLength"].max()
    return gl


def counts2log1tpm(adata, genelen_file=None):

    if genelen_file is None:
        with importlib.resources.path("PopCILA.data", "GeneLength.txt") as default_path:
            genelen_file = default_path

    if isinstance(genelen_file, Path):
        genelen_file = str(genelen_file)

    gl = _load_gene_length(genelen_file)

    adata = adata[:, ~adata.var_names.duplicated()].copy()

    inter = np.intersect1d(adata.var_names, gl.index)
    if len(inter) == 0:
        raise ValueError("No overlapping genes found between adata.var_names and gene length data.")

    adata = adata[:, inter].copy()
    gl = gl.loc[inter].astype(np.float32)

    len_kb = gl.values / 1000.0  # shape: (n_genes,)

    X = adata.X
    if not issparse(X):
        X = csr_matrix(X)
    X = X.astype(np.float32)

    inv_len = 1.0 / len_kb  # (n_genes,)
    X_len_norm = X.multiply(inv_len)  # counts / len_kb

    row_sum = np.asarray(X_len_norm.sum(axis=1)).ravel().astype(np.float32)

    scale = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum > 0
    scale[nonzero] = 1e6 / row_sum[nonzero]  

    X_tpm = X_len_norm.multiply(scale.reshape(-1, 1))

    X_tpm = X_tpm.tocsr().astype(np.float32)
    X_tpm.data = np.log1p(X_tpm.data)  # log1(TPM + 1)

    adata.X = X_tpm
    adata.uns["log1p"] = {"base": None}

    return adata


def select_z_score_lr(H_array,comm_matrix,threshold=2):
    df = pd.DataFrame(H_array,index = comm_matrix.index,columns=['loading'])
    df_filtered = df[df['loading'] != 0].copy()
    df_filtered['z_score'] = zscore(df_filtered['loading'])
    return df_filtered[df_filtered['z_score'] > threshold]

@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
	sum=0
	for i in range(t1.shape[0]):
		sum+=(t1[i]-t2[i])**2
	return np.sqrt(sum)

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
	n=X.shape[0]
	adj=np.empty((n, n), dtype=np.float32)
	for i in numba.prange(n):
		for j in numba.prange(n):
			adj[i][j]=euclid_dist(X[i], X[j])
	return adj

def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
	#x,y,x_pixel, y_pixel are lists
	if histology:
		assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
		assert (len(x)==len(x_pixel)) & (len(y)==len(y_pixel))
		print("Calculateing adj matrix using histology image...")
		#beta to control the range of neighbourhood when calculate grey vale for one spot
		#alpha to control the color scale
		beta_half=round(beta/2)
		g=[]
		for i in range(len(x_pixel)):
			max_x=image.shape[0]
			max_y=image.shape[1]
			nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
			g.append(np.mean(np.mean(nbs,axis=0),axis=0))
		c0, c1, c2=[], [], []
		for i in g:
			c0.append(i[0])
			c1.append(i[1])
			c2.append(i[2])
		c0=np.array(c0)
		c1=np.array(c1)
		c2=np.array(c2)
		c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
		c4=(c3-np.mean(c3))/np.std(c3)
		z_scale=np.max([np.std(x), np.std(y)])*alpha
		z=c4*z_scale
		z=z.tolist()
		X=np.array([x, y, z]).T.astype(np.float32)
	else:
		print("Calculateing adj matrix using xy only...")
		X=np.array([x, y]).T.astype(np.float32)
	return pairwise_distance(X)

def create_knn_adj(adj, k=20):
    """
    Construct an adjacency graph using the k-nearest neighbors algorithm and return a sparse matrix.

    Parameters:
    adj (np.ndarray or scipy.sparse matrix): Input adjacency matrix.
    k (int): Number of nearest neighbors, default is 20.

    Returns:
    adj_sparse (scipy.sparse.csr_matrix): Processed sparse adjacency matrix.
    """
    adj_sparse = kneighbors_graph(adj, k, mode='connectivity')

    return adj_sparse

def spatial_knn_graph(
    x,
    y,
    x_pixel=None,
    y_pixel=None,
    image=None,
    histology=False,
    beta=49,
    alpha=1.0,
    n_neighbors=20,
    mode="connectivity",      # "connectivity" or "distance"
    symmetrize=True,          # Whether to symmetrize the adjacency matrix
    n_jobs=-1,
    return_features=False,
):
    """
    Construct a sparse k-nearest-neighbor adjacency matrix from (x, y) or (x, y, z) features.
    
    Parameters
    x, y : 1D array-like
        Spatial coordinates of spots on the slide (usually x/y positions).
    x_pixel, y_pixel : 1D array-like, optional
        Pixel coordinates in the H&E image corresponding to x, y.
        Required only when histology=True.
    image : np.ndarray, optional
        H&E RGB image of shape (H, W, 3) and dtype uint8 or float.
        Required only when histology=True.
    histology : bool, default False
        Whether to incorporate image information to build a third feature z:
        - False: use (x, y) only
        - True: extract local color features from the image and build (x, y, z)
    beta : int, default 49
        Side length (in pixels) of the local patch extracted around each spot; a beta×beta window is used.
    alpha : float, default 1.0
        Scaling factor for the z direction: z_scale = max(std(x), std(y)) * alpha
    n_neighbors : int, default 20
        Number of nearest neighbors k for each spot.
    mode : {"connectivity", "distance"}, default "connectivity"
        Type of adjacency matrix:
        - "connectivity": 0/1 adjacency
        - "distance": edge weights are Euclidean distances
    symmetrize : bool, default True
        Whether to symmetrize the graph (recommended True for undirected graph).
    n_jobs : int, default -1
        Number of parallel threads passed to sklearn.neighbors.kneighbors_graph.
    return_features : bool, default False
        If True, return (adj, X); otherwise return adj only.
    
    Returns
    adj : scipy.sparse.csr_matrix, shape (n_spots, n_spots)
        Sparse k-nearest-neighbor adjacency matrix.
    X : np.ndarray, shape (n_spots, d), optional
        Feature matrix used (d=2 or 3), returned only when return_features=True.
    """

    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    n = x.shape[0]

    k = min(int(n_neighbors), n - 1)

    if histology:
        if x_pixel is None or y_pixel is None or image is None:
            raise ValueError("When histology=True, x_pixel, y_pixel and image must be provided.")

        x_pixel = np.asarray(x_pixel, dtype=int).ravel()
        y_pixel = np.asarray(y_pixel, dtype=int).ravel()
        assert x_pixel.shape == x.shape and y_pixel.shape == y.shape, \
            "The length of x_pixel / y_pixel must be the same as that of x / y."

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("The image must be an RGB image with a shape of (H, W, 3)")

        print("Building feature matrix [x, y, z] using histology image...")

        beta_half = int(round(beta / 2))
        H, W, _ = image.shape
        g_list = []

        for xv, yv in zip(x_pixel, y_pixel):
            x0 = max(0, xv - beta_half)
            x1 = min(H, xv + beta_half + 1)
            y0 = max(0, yv - beta_half)
            y1 = min(W, yv + beta_half + 1)
            patch = image[x0:x1, y0:y1]      # (h, w, 3)
            g_list.append(patch.mean(axis=(0, 1)))  # RGB mean

        g = np.asarray(g_list, dtype=np.float32)   # (n, 3)
        c0, c1, c2 = g[:, 0], g[:, 1], g[:, 2]

        var0, var1, var2 = np.var(c0), np.var(c1), np.var(c2)
        denom = var0 + var1 + var2 if (var0 + var1 + var2) > 0 else 1.0
        c3 = (c0 * var0 + c1 * var1 + c2 * var2) / denom

        c3_mean = c3.mean()
        c3_std = c3.std() if c3.std() > 0 else 1.0
        c4 = (c3 - c3_mean) / c3_std

        z_scale = max(np.std(x), np.std(y)) * float(alpha)
        z = c4 * z_scale

        X = np.vstack([x, y, z]).T.astype(np.float32)  # (n, 3)
    else:
        print("Building feature matrix [x, y] only...")
        X = np.vstack([x, y]).T.astype(np.float32)      # (n, 2)

    adj = kneighbors_graph(
        X,
        n_neighbors=k,
        mode=mode,          # "connectivity" 或 "distance"
        metric="euclidean",
        include_self=False,
        n_jobs=n_jobs,
    )

    adj = adj.tocsr().astype(np.float32)

    if symmetrize:
        if mode == "connectivity":
            adj = adj.maximum(adj.T)
        else:  # mode == "distance"
            adj = adj.minimum(adj.T)

    if return_features:
        return adj, X
    else:
        return adj


def getLRcomm(W, H,comm_matrix, patterns,lr_expr_pairs,position = None,zscore_threshold = 2):
    phe_factor_sampleW = {}
    phe_factor_cci = {}
    phe_factor_cciW = {}
    for i in patterns:
        if position is not None:
            phe_factor_sampleW[i] = np.mean(W[:, i-1][position[0]:position[1]+1])
        else:
            phe_factor_sampleW[i] = np.mean(W[:, i-1])
        phe_factor_cci[i] = select_z_score_lr(H[i-1],comm_matrix,threshold=zscore_threshold)
        phe_factor_cciW[i] = phe_factor_sampleW[i] * phe_factor_cci[i]
    final_df = merge_loading_dfs(phe_factor_cciW)
    final_df = final_df.reset_index().rename(columns={'index': 'l-r'}) 
    
    result = pd.merge(final_df, lr_expr_pairs[lr_expr_pairs['l-r'].isin(final_df['l-r'])], left_on='l-r', right_on='l-r', how='inner')
    result = result[['l-r','ligand','receptor','loading']]
    
    return result

def getLRTFcomm(W, H,comm_matrix, patterns,lr_expr_pairs,position = None,zscore_threshold = 2):
    phe_factor_sampleW = {}
    phe_factor_cci = {}
    phe_factor_cciW = {}
    for i in patterns:
        if position is not None:
            phe_factor_sampleW[i] = np.mean(W[:, i-1][position[0]:position[1]+1])
        else:
            phe_factor_sampleW[i] = np.mean(W[:, i-1])
        phe_factor_cci[i] = select_z_score_lr(H[i-1],comm_matrix,threshold=zscore_threshold)
        phe_factor_cciW[i] = phe_factor_sampleW[i] * phe_factor_cci[i]
    final_df = merge_loading_dfs(phe_factor_cciW)
    final_df = final_df.reset_index().rename(columns={'index': 'l-r-tf'}) 
    
    result = pd.merge(final_df, lr_expr_pairs[lr_expr_pairs['l-r-tf'].isin(final_df['l-r-tf'])], left_on='l-r-tf', right_on='l-r-tf', how='inner')
    result = result[['l-r-tf','ligand','receptor','TF','loading']]
    
    return result

def merge_loading_dfs(df_dict):
    all_indexes = pd.Index([])  
    for df in df_dict.values():
        all_indexes = all_indexes.union(df.index)  
    final_df = pd.DataFrame(index=all_indexes)

    for df in df_dict.values():
        final_df['loading'] = final_df.get('loading', 0)  
        final_df['loading'] += df['loading'].reindex(final_df.index, fill_value=0)

    return final_df.sort_values(by='loading', ascending=False)

def extract_lr_tf(df: pd.DataFrame):
    cols = set(df.columns)

    if {"ligand", "receptor", "TF"}.issubset(cols):
        return list(df[["ligand", "receptor", "TF"]].itertuples(index=False, name=None))

    if {"ligand", "receptor"}.issubset(cols):
        return list(df[["ligand", "receptor"]].itertuples(index=False, name=None))

    return []


def Singleassociationanalysis(W, clin_df, phenotype_types, covariates=None, alpha=0.05):
    """
    Perform regression analysis between the NMF pattern weight matrix W and multiple phenotypes (For single element).

    Parameters:
    - W: np.ndarray or pd.DataFrame, (samples, patterns) NMF pattern weight matrix
    - clin_df: pd.DataFrame, (samples, phenotypes) DataFrame containing patient phenotype data
    - phenotype_types: dict, specifying the variable type for each phenotype {"phenotype_name": "binary" or "continuous" or "ordinal"}
    - covariates: pd.DataFrame or None, (samples, covariate_num) Optional covariates (e.g., age, gender)
    - alpha: float, default 0.05, significance threshold after multiple correction

    Returns:
    - results_df: pd.DataFrame, regression analysis results, including regression coefficients, P-values, and FDR-adjusted P-values
    """
    if isinstance(W, np.ndarray):
        W = pd.DataFrame(W, columns=[f"Pattern_{i+1}" for i in range(W.shape[1])], index=clin_df.index)

    results = []

    # Iterate over each phenotype
    for phe in list(phenotype_types.keys()):
        if phe not in clin_df.columns:
            print(f"Warning: {phe} not in clin_df 中, skipping this variable.")
            continue

        # dropna
        valid_idx = clin_df[phe].dropna().index
        Y = clin_df.loc[valid_idx, phe]
        W_valid = W.loc[valid_idx, :]
        cov_valid = covariates.loc[valid_idx, :] if covariates is not None else None

        # Iterate each patterns
        for mode in W_valid.columns:
            X = W_valid[[mode]]  
            if cov_valid is not None:
                X = pd.concat([X, cov_valid], axis=1)

            if phenotype_types[phe] == "continuous":
                X = sm.add_constant(X)  
                model = sm.OLS(Y, X).fit()
                coef = model.params.iloc[1]
                p_value = model.pvalues.iloc[1]
                r_squared = model.rsquared
            elif phenotype_types[phe] == "binary":
                X = sm.add_constant(X)  
                model = sm.Logit(Y, X).fit(disp=0)
                coef = model.params.iloc[1]
                p_value = model.pvalues.iloc[1]
                r_squared = model.prsquared
            elif phenotype_types[phe] == "ordinal":
                Y = Y.astype("category") 
                Y = Y.cat.set_categories(sorted(Y.unique()), ordered=True)
                model = OrderedModel(Y, X, distr="logit").fit(method="bfgs", disp=False)
                coef = model.params.iloc[0]
                p_value = model.pvalues.iloc[0]
                r_squared = None

            else:
                print(f"Error: Unsupported phenotype type {phenotype_types[phe]}，Skipping {phe}.")
                continue

            results.append({"Phenotype": phe, "Mode": mode, "Coefficient": coef, "P_value": p_value, "R_squared": r_squared})

    results_df = pd.DataFrame(results)

    # **FDR **
    if not results_df.empty:
        results_df["Adjusted_P_value"] = np.nan  

        for phe in results_df["Phenotype"].unique():
            phe_mask = results_df["Phenotype"] == phe
            raw_p_values = results_df.loc[phe_mask, "P_value"].values
            _, corrected_p_values, _, _ = multipletests(raw_p_values, method="fdr_bh")
            results_df.loc[phe_mask, "Adjusted_P_value"] = corrected_p_values

    else:
        print("Not enough data for regression analysis.")
        return None
    return results_df

def Covariateassociationanalysis(W, clin_df, phenotype_types, covariates, alpha=0.05):
    """
    Perform multivariate regression analysis between the NMF pattern weight matrix W and clinical phenotypes (with covariates).

    Parameters:
    - W: np.ndarray or pd.DataFrame, (samples, patterns) NMF pattern weight matrix
    - clin_df: pd.DataFrame, (samples, features) Clinical data containing phenotypes and covariates
    - phenotype_types: dict, specifying the types of phenotypes of interest (continuous, binary, ordinal)
    - covariates: dict, specifying the covariates and their types (continuous, binary, ordinal)
    - alpha: float, default 0.05, significance threshold after multiple correction

    Returns:
    - results_df: pd.DataFrame, multivariate regression results, including regression coefficients, P-values, and FDR-adjusted P-values
    """
    if isinstance(W, np.ndarray):
        W = pd.DataFrame(W, columns=[f"Pattern_{i+1}" for i in range(W.shape[1])],index=clin_df.index)

    results = []

    for phenotype, p_type in phenotype_types.items():
        if phenotype not in clin_df.columns:
            continue  
        

        relevant_cols = [phenotype] + covariates
        df = clin_df.join(W, how="inner")  
        df = df[relevant_cols + list(W.columns)].dropna().copy()  

        if df.empty:
            continue

        for mode in W.columns:
            X = df[[mode] + covariates]  
            Y = df[phenotype]  
            X = sm.add_constant(X)  

            try:
                if p_type == "continuous":
                    model = sm.OLS(Y, X).fit()
                elif p_type == "binary":
                    model = Logit(Y, X).fit(disp=0)
                elif p_type == "ordinal":
                    model = OrderedModel(Y, X, distr="logit").fit(method="bfgs", disp=0)
                else:
                    raise ValueError(f"Unsupported phenotype type: {p_type}")

                coef = model.params.iloc[1]  
                p_value = model.pvalues.iloc[1]  

                results.append({"Phenotype": phenotype, "Mode": mode, "Coefficient": coef, "P_value": p_value})

            except Exception as e:
                print(f"Error in model fitting for {phenotype} - {mode}: {e}")
                continue


    results_df = pd.DataFrame(results)

    # FDR
    if not results_df.empty:
        for phenotype in phenotype_types.keys():
            mask = results_df["Phenotype"] == phenotype
            if mask.sum() > 0:
                _, adj_p_values, _, _ = multipletests(results_df.loc[mask, "P_value"], method="fdr_bh")
                results_df.loc[mask, "Adjusted_P_value"] = adj_p_values
    else:
        print("Not enough data for regression analysis.")
        return None

    return results_df

"""
def plot_significance_heatmap(results_df, alpha, savefig=None):

    Plot a significance heatmap using -log10(Adjusted P value) to display significance.
 

    results_df["log10_p_adjust"] = -np.log10(results_df["Adjusted_P_value"])

    heatmap_data = results_df.pivot(index="Phenotype", columns="Mode", values="log10_p_adjust")
    significance = results_df.pivot(index="Phenotype", columns="Mode", values="Adjusted_P_value") < alpha

    significance_labels = significance.replace({True: '*', False: ''})
    
    plt.figure(figsize=(10, 1.5))
    ax = sns.heatmap(heatmap_data, annot=significance_labels, fmt="",cmap="coolwarm", center=0, linewidths=0.5,
        annot_kws={"fontsize": 12, "color": "black"},  
        cbar_kws={"shrink": 0.8}  )
    
    plt.title("Significance Heatmap of NMF Patterns & Phenotypes (-log10(Adjusted P))")
    plt.xlabel("NMF Pattern")
    plt.ylabel("Phenotype")
    
    cbar = ax.collections[0].colorbar
    cbar.set_label("-log10(Adjusted P value)")
    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()
"""

def plot_significance_heatmap(results_df, alpha=0.05, savefig=None):
    """
    Plot a significance heatmap using -log10(Adjusted P value) to display significance.
    
    Parameters:
    - results_df: DataFrame containing 'Phenotype', 'Mode', 'Adjusted_P_value', and 'Coefficient'.
    - alpha: Significance threshold for adjusted P-values.
    - savefig: If provided, saves the figure to the specified path.
    """
    results_df["log10_p_adjust"] = -np.log10(results_df["Adjusted_P_value"])
    
    results_df["Color"] = np.where(results_df["Coefficient"] > 0, results_df["log10_p_adjust"], -results_df["log10_p_adjust"])

    heatmap_data = results_df.pivot(index="Phenotype", columns="Mode", values="Color")
    significance = results_df["Adjusted_P_value"] < alpha

    significance_labels = results_df.pivot(index="Phenotype", columns="Mode", values="Adjusted_P_value")
    significance_labels = significance_labels < alpha
    significance_labels = significance_labels.replace({True: '*', False: ''})

    num_phenotypes = len(heatmap_data)
    figsize_height = 1.5 * num_phenotypes
    
    plt.figure(figsize=(10, figsize_height))
    ax = sns.heatmap(heatmap_data, annot=significance_labels, fmt="", cmap="bwr", center=0, linewidths=0.5,
                     annot_kws={"fontsize": 12, "color": "black"}, cbar_kws={"shrink": 0.8})
    
    plt.title("Significance Heatmap of Patterns & Phenotypes\n (Red: Positive, Blue: Negative)")
    plt.xlabel("Pattern")
    plt.ylabel("Phenotype")
    
    cbar = ax.collections[0].colorbar
    cbar.set_label("-log10(P value)")
    
    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sankey(
    res, 
    title="Ligand-Receptor", 
    width=1000, 
    height=800, 
    pad=100, 
    thickness=40, 
    font_size=15, 
    node_colors=None, 
    link_colors=None, 
    line_color="black", 
    line_width=0.5
):
    """
    Generate an interactive Sankey diagram for Ligand-Receptor interactions.

    Parameters:
    res (pd.DataFrame): DataFrame containing columns 'ligand', 'receptor', and 'loading'.
    title (str): Title of the diagram, default is "Ligand-Receptor".
    width (int): Width of the diagram, default is 1000.
    height (int): Height of the diagram, default is 800.
    pad (int): Spacing between nodes, default is 100.
    thickness (int): Thickness of the nodes, default is 40.
    font_size (int): Font size, default is 15.
    node_colors (list): List of colors for nodes, default is None (blue for ligands, green for receptors).
    link_colors (list): List of colors for links, default is None (semi-transparent green).
    line_color (str): Border color of nodes, default is "black".
    line_width (float): Border width of nodes, default is 0.5.

    Returns:
    HTML: HTML object of the interactive Sankey diagram.
    """
    ligands = res['ligand'].unique()
    receptors = res['receptor'].unique()

    nodes = list(ligands) + list(receptors)

    node_indices = {node: i for i, node in enumerate(nodes)}

    source = []
    target = []
    value = []

    for _, row in res.iterrows():
        source.append(node_indices[row['ligand']])  # ligand indexs
        target.append(node_indices[row['receptor']])  # receptor indexs
        value.append(row['loading'])  # 权重

    if node_colors is None:
        node_colors = ["#7495D3"] * len(ligands) + ["#C798EE"] * len(receptors)

    if link_colors is None:
        link_colors = ["#D1D1D1"] * len(source) 


    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=pad,  
            thickness=thickness,  
            line=dict(color=line_color, width=line_width), 
            label=nodes,
            color=node_colors  
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors  
        )
    )])


    fig.update_layout(
        title_text=title,
        font_size=font_size,  
        width=width,  
        height=height  
    )

    return HTML(fig.to_html())

def plot_sankey_LRTF(
    res,
    title="Ligand-Receptor-TF",
    width=1200,
    height=800,
    pad=100,
    thickness=40,
    font_size=15,
    node_colors=None,
    link_colors=None,
    line_color="black",
    line_width=0.5
):
    """
    Generate an interactive Sankey diagram for Ligand-Receptor-TF interactions.

    Parameters:
    res (pd.DataFrame): DataFrame containing columns ['ligand', 'receptor', 'TF', 'loading'].
    title (str): Title of the diagram, default "Ligand-Receptor-TF".
    width (int): Width of the diagram.
    height (int): Height of the diagram.
    pad (int): Spacing between nodes.
    thickness (int): Node thickness.
    font_size (int): Font size for labels.
    node_colors (list): Optional list of colors for nodes. Default: blue (ligands), green (receptors), purple (TFs).
    link_colors (list): Optional list of colors for links.
    line_color (str): Border color for nodes.
    line_width (float): Border width for nodes.

    Returns:
    HTML: Interactive Sankey diagram.
    """
    # === Unique nodes by group ===
    ligands = res['ligand'].unique()
    receptors = res['receptor'].unique()
    tfs = res['TF'].unique()

    # Combine all nodes (L, R, TF)
    nodes = list(ligands) + list(receptors) + list(tfs)
    node_indices = {node: i for i, node in enumerate(nodes)}

    # === Links (two layers: L→R and R→TF) ===
    source, target, value, link_color = [], [], [], []

    # L → R
    for _, row in res.iterrows():
        s = node_indices[row['ligand']]
        t = node_indices[row['receptor']]
        source.append(s)
        target.append(t)
        value.append(row['loading'])
        link_color.append("rgba(100, 149, 237, 0.4)")  # light blue

    # R → TF
    for _, row in res.iterrows():
        s = node_indices[row['receptor']]
        t = node_indices[row['TF']]
        source.append(s)
        target.append(t)
        value.append(row['loading'])
        link_color.append("rgba(199, 152, 238, 0.4)")  # light purple

    # === Node colors ===
    if node_colors is None:
        ligand_colors = ["#7495D3"] * len(ligands)    
        receptor_colors = ["#C798EE"] * len(receptors) 
        tf_colors = ["#8FD694"] * len(tfs)            
        node_colors = ligand_colors + receptor_colors + tf_colors

    # === Link colors ===
    if link_colors is None:
        link_colors = link_color

    # === Plotly Sankey ===
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=pad,
            thickness=thickness,
            line=dict(color=line_color, width=line_width),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])

    fig.update_layout(
        title_text=title,
        font_size=font_size,
        width=width,
        height=height
    )

    return HTML(fig.to_html())

def calculate_correlation_matrix( 
    bulk_lr_expr_df,
    sc_lr_expr_df,
    block_size=2000,
    dtype=np.float32,
):
    common_genes = sc_lr_expr_df.index.intersection(bulk_lr_expr_df.index)
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between bulk and sc.")

    bulk_lr_expr_df = bulk_lr_expr_df.loc[common_genes].astype(dtype)
    sc_lr_expr_df   = sc_lr_expr_df.loc[common_genes].astype(dtype)

    n_bulk_samples = bulk_lr_expr_df.shape[1]
    n_cell_samples = sc_lr_expr_df.shape[1]

    dataset0 = np.concatenate(
        [bulk_lr_expr_df.values, sc_lr_expr_df.values],
        axis=1
    ).astype(dtype)

    del bulk_lr_expr_df, sc_lr_expr_df
    gc.collect()

    # QN
    data_qn = qnorm.quantile_normalize(dataset0, axis=1).astype(dtype)

    del dataset0
    gc.collect()

    n_genes = data_qn.shape[0]

    Expression_bulk = data_qn[:, :n_bulk_samples]    # (G, n_bulk)
    Expression_cell = data_qn[:, n_bulk_samples:]    # (G, n_cell)

    # bulk
    bulk_mean = Expression_bulk.mean(axis=0, keepdims=True)
    bulk_std  = Expression_bulk.std(axis=0, ddof=1, keepdims=True)
    bulk_std[bulk_std == 0] = 1e-6
    
    Expression_bulk -= bulk_mean
    Expression_bulk /= bulk_std
    
    # cell
    cell_mean = Expression_cell.mean(axis=0, keepdims=True)
    cell_std  = Expression_cell.std(axis=0, ddof=1, keepdims=True)
    cell_std[cell_std == 0] = 1e-6
    
    Expression_cell -= cell_mean
    Expression_cell /= cell_std
    
    del bulk_mean, bulk_std, cell_mean, cell_std
    gc.collect()
    
    X = np.empty((n_bulk_samples, n_cell_samples), dtype=dtype)
    G_minus_1 = float(n_genes - 1)
    
    bulk_z_T = Expression_bulk.T
    
    for j_start in range(0, n_cell_samples, block_size):
        j_end = min(n_cell_samples, j_start + block_size)
        cell_block = Expression_cell[:, j_start:j_end]
    
        corr_block = (bulk_z_T @ cell_block) / G_minus_1
        X[:, j_start:j_end] = corr_block


    del data_qn, Expression_bulk, Expression_cell, bulk_z_T
    gc.collect()

    quality_check = np.percentile(X, [0, 25, 50, 75, 100])

    print("|**************************************************|")
    print("Performing quality-check for the correlations")
    print("The five-number summary of correlations:")
    print(f"Min: {quality_check[0]}")
    print(f"25th Percentile: {quality_check[1]}")
    print(f"Median: {quality_check[2]}")
    print(f"75th Percentile: {quality_check[3]}")
    print(f"Max: {quality_check[4]}")
    print("|**************************************************|")

    if quality_check[2] < 0.1:
        print("Warning: The median correlation between the single-cell and bulk samples is relatively low.")

    return X

def calculate_LTF_correlation_matrix(bulk_lr_expr_df, sc_lr_expr_df, ligand_tf_df):
    """
    Calculate the strictly weighted equal-pair Pearson correlation matrix between
    bulk and single-cell expression data. Each ligand–TF pair contributes equally.
    This implementation uses weighted mean and weighted std dev for standardization,
    which is the textbook definition of weighted Pearson correlation.

    Parameters
    ----------
    bulk_lr_expr_df : pd.DataFrame
        Expression matrix of bulk data (rows = genes, columns = samples).
    sc_lr_expr_df : pd.DataFrame
        Expression matrix of single-cell data (rows = genes, columns = samples).
    ligand_tf_df : pd.DataFrame
        DataFrame with columns ['ligand', 'TF'], mapping ligands to TFs.

    Returns
    -------
    X_equalpair : np.ndarray
        Pair-based (equal-pair) strictly weighted Pearson correlation matrix
        between bulk and single-cell data. Values are guaranteed to be in [-1, 1].
    """

    # 1. Align genes and prepare data
    common_genes = bulk_lr_expr_df.index.intersection(sc_lr_expr_df.index)
    bulk_expr = bulk_lr_expr_df.loc[common_genes]
    sc_expr = sc_lr_expr_df.loc[common_genes]

    dataset = pd.concat([bulk_expr, sc_expr], axis=1).astype(np.float64)
    dataset_qn = qnorm.quantile_normalize(dataset, axis=1)
    dataset_qn = pd.DataFrame(dataset_qn, index=dataset.index, columns=dataset.columns)

    n_bulk = bulk_expr.shape[1]
    Expression_bulk = dataset_qn.iloc[:, :n_bulk]
    Expression_cell = dataset_qn.iloc[:, n_bulk:]

    # 2. Construct pair-based equal weights
    n_pairs = len(ligand_tf_df)
    if n_pairs == 0:
        return np.array([[]]) # Handle case with no pairs
        
    w_pair = 1.0 / n_pairs
    gene_weights = pd.Series(0.0, index=dataset_qn.index)
    for _, row in ligand_tf_df.iterrows():
        ligand, tf = row['ligand'], row['TF']
        if ligand in gene_weights.index:
            gene_weights[ligand] += w_pair / 2
        if tf in gene_weights.index:
            gene_weights[tf] += w_pair / 2

    # Normalize weights to sum to 1
    total_weight = gene_weights.sum()
    if total_weight > 0:
        gene_weights /= total_weight
    
    W = gene_weights.values.reshape(-1, 1) # Shape: (n_genes, 1)

    # 3. Perform weighted Z-score standardization for each sample
    bulk_mean_w = np.dot(Expression_bulk.T, W) # Shape: (n_bulk, 1)
    cell_mean_w = np.dot(Expression_cell.T, W) # Shape: (n_cell, 1)

    bulk_centered = Expression_bulk - bulk_mean_w.T
    cell_centered = Expression_cell - cell_mean_w.T

    # Weighted standard deviation: sqrt(sum(w * (x - mean_w)**2))
    bulk_std_w = np.sqrt(np.dot(bulk_centered.pow(2).T, W)) # Shape: (n_bulk, 1)
    cell_std_w = np.sqrt(np.dot(cell_centered.pow(2).T, W)) # Shape: (n_cell, 1)
    
    # Avoid division by zero
    bulk_std_w[bulk_std_w < 1e-10] = 1.0
    cell_std_w[cell_std_w < 1e-10] = 1.0

    bulk_z = bulk_centered / bulk_std_w.T
    cell_z = cell_centered / cell_std_w.T

    # 4. Calculate the weighted correlation matrix
    # This is the weighted covariance of the weighted-Z-scored data.
    # Since the data is properly scaled, this equals the correlation.
    # Formula: Cor(B_j, C_k) = sum_i(w_i * B_z_{ij} * C_z_{ik})
    # Matrix form: B_z.T @ (W * C_z)
    X_equalpair = np.dot(bulk_z.values.T, W * cell_z.values)

    # 5. Quality Check
    quality_check = np.percentile(X_equalpair, [0, 25, 50, 75, 100])
    print("|**************************************************|")
    print("Performing equal-pair correlation quality-check")
    print("The five-number summary of equal-pair correlations:")
    print(f"Min: {quality_check[0]:.4f}")
    print(f"25th Percentile: {quality_check[1]:.4f}")
    print(f"Median: {quality_check[2]:.4f}")
    print(f"75th Percentile: {quality_check[3]:.4f}")
    print(f"Max: {quality_check[4]:.4f}")
    print("|**************************************************|")

    # The check for max > 1 is no longer needed with the corrected formula.
    if np.any(X_equalpair > 1.0001) or np.any(X_equalpair < -1.0001):
        print("Warning: Correlation is outside the [-1, 1] range. Check the calculation.")
        
    if quality_check[2] < 0.1:
        print("Warning: Median correlation is relatively low. Check data quality or normalization.")

    return X_equalpair

def similarity2adjacent(adata, key=None):
    """
    Calculate the cell adjacency matrix based on the cell similarity sparse matrix in adata.obsp.

    Parameters:
    adata (AnnData): AnnData object.
    key (str): Key in adata.obsp.

    Returns:
    connectivities (scipy.sparse.csr_matrix): Cell adjacency matrix derived from the cell similarity sparse matrix.
    """
    connectivities = adata.obsp[key]

    connectivities.setdiag(0)

    connectivities.data = np.ones_like(connectivities.data)

    return connectivities


def center_features(X):
    feature_means = X.mean(dim=0, keepdim=True)
    X_centered = X - feature_means
    return X_centered, feature_means


def getPosipotentialCCI(
    adata,
    model,
    threshold_percent=60,
    spot_size=None,
    stability_threshold=None,        
    bootstrap_threshold=None,        
    savefig=None,
    embedding='umap'
):
    if hasattr(model, "linear"):
        beta = model.linear.weight.detach().cpu().squeeze().numpy()
        stability_pos = None
        bootstrap_pos = None
    elif hasattr(model, "beta"):
        beta = np.asarray(model.beta)
        stability_pos = getattr(model, "stability_pos", None)
        bootstrap_pos = getattr(model, "bootstrap_support_pos", None)
    else:
        raise ValueError("model must be a PyTorch model or an instance of CCcommInfer")

    threshold = np.percentile(beta, threshold_percent)

    mask = (beta > 0) & (beta >= threshold)

    #title_parts = [f"top {100 - threshold_percent}% β"]
    title_parts = []

    if (stability_threshold is not None) and (stability_pos is not None):
        stability_pos = np.asarray(stability_pos)
        mask = mask & (stability_pos >= stability_threshold)
        title_parts.append(f"stability_pos≥{stability_threshold}")

    if (bootstrap_threshold is not None) and (bootstrap_pos is not None):
        bootstrap_pos = np.asarray(bootstrap_pos)
        mask = mask & (bootstrap_pos >= bootstrap_threshold)
        title_parts.append(f"bootstrap_pos≥{bootstrap_threshold}")

    marked = np.where(mask, 1, 0)

    adata.obs['phe_cell'] = marked.astype(str)
    highlight_palette = {'0': 'lightgrey', '1': 'red'}

    if f'X_{embedding}' not in adata.obsm:
        raise ValueError(
            f"Embedding '{embedding}' not found in adata.obsm. "
            f"Please run the corresponding dimensionality reduction first."
        )

    #title = 'Positive phenotype-associated CCI cells\n' + ", ".join(title_parts) 
    title = 'Phenotype associated CCI active cells'

    if spot_size is not None:
        fig = sc.pl.embedding(
                adata,
                basis=embedding,
                color='phe_cell',
                palette=highlight_palette,
                show=False,
                return_fig=True,
                title=title,
                size=spot_size
        )
    else:
        fig = sc.pl.embedding(
                    adata,
                    basis=embedding,
                    color='phe_cell',
                    palette=highlight_palette,
                    show=False,
                    return_fig=True,
                    title=title,
        )

    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {savefig}")

    plt.show()


def getNegapotentialCCI(
    adata,
    model,
    threshold_percent=40,
    spot_size=None,
    stability_threshold=None,        
    bootstrap_threshold=None,        
    savefig=None,
    embedding='umap'
):

    if hasattr(model, "linear"):
        beta = model.linear.weight.detach().cpu().squeeze().numpy()
        stability_neg = None
        bootstrap_neg = None
    elif hasattr(model, "beta"):
        beta = np.asarray(model.beta)
        stability_neg = getattr(model, "stability_neg", None)
        bootstrap_neg = getattr(model, "bootstrap_support_neg", None)
    else:
        raise ValueError("model must be a PyTorch model or an instance of CCcommInfer")

    negative_beta = beta[beta < 0]

    threshold = np.percentile(negative_beta, threshold_percent)

    mask = (beta < 0) & (beta <= threshold)

    #title_parts = [f"most negative {threshold_percent}% β"]
    title_parts = []

    if (stability_threshold is not None) and (stability_neg is not None):
        stability_neg = np.asarray(stability_neg)
        mask = mask & (stability_neg >= stability_threshold)
        title_parts.append(f"stability_neg≥{stability_threshold}")

    if (bootstrap_threshold is not None) and (bootstrap_neg is not None):
        bootstrap_neg = np.asarray(bootstrap_neg)
        mask = mask & (bootstrap_neg >= bootstrap_threshold)
        title_parts.append(f"bootstrap_neg≥{bootstrap_threshold}")

    marked = np.where(mask, 1, 0)

    adata.obs['phe_cell'] = marked.astype(str)
    highlight_palette = {'0': 'lightgrey', '1': 'blue'}

    if f'X_{embedding}' not in adata.obsm:
        raise ValueError(
            f"Embedding '{embedding}' not found in adata.obsm. "
            f"Please run the corresponding dimensionality reduction first."
        )

    #title = 'Negative phenotype-associated CCI cells\n' + ", ".join(title_parts)
    title = 'Phenotype associated CCI active cells'

    if spot_size is not None:
        fig = sc.pl.embedding(
                adata,
                basis=embedding,
                color='phe_cell',
                palette=highlight_palette,
                show=False,
                return_fig=True,
                title=title,
                size=spot_size
        )
    else:
        fig = sc.pl.embedding(
                    adata,
                    basis=embedding,
                    color='phe_cell',
                    palette=highlight_palette,
                    show=False,
                    return_fig=True,
                    title=title,
        )

    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {savefig}")

    plt.show()


def getCCcomm(adata, gene1, gene2, L_threshold=20, R_threshold=20, marked_col='phe_cell', 
             savefig=None, saveflag=True, size_factor=None, embedding='umap'):
    """
    Mark cells with high expression of two genes on a specified embedding plot, limited to cells where marked is 1.

    Parameters:
    - adata: AnnData object.
    - gene1: Name of the first gene (e.g., Ligand).
    - gene2: Name of the second gene (e.g., Receptor).
    - L_threshold: High expression percentile threshold for the first gene (default is top 20%).
    - R_threshold: High expression percentile threshold for the second gene (default is top 20%).
    - marked_col: Column name for the marked column.
    - savefig: Filename to save the figure.
    - saveflag: Whether to delete the temporary annotation after plotting.
    - size_factor: Scaling factor for point sizes. If None, default sizes are used.
    - embedding: Embedding to use for plotting (e.g., 'umap', 'tsne', 'pca'). Default is 'umap'.
    """

    if f'X_{embedding}' not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm. Please run {embedding.upper()} first.")
    
    marked_cells = adata.obs[marked_col] == '1' if marked_col else np.ones(adata.n_obs, dtype=bool)
    
    gene1_expression = adata[:, gene1].X.toarray().flatten()
    gene2_expression = adata[:, gene2].X.toarray().flatten()
    
    gene1_threshold = np.percentile(gene1_expression, 100 - L_threshold)
    gene2_threshold = np.percentile(gene2_expression, 100 - R_threshold)
    
    gene1_high = (gene1_expression >= gene1_threshold) & (gene1_expression > 0)
    gene2_high = (gene2_expression >= gene2_threshold) & (gene2_expression > 0)
    
    adata.obs[f'{gene1}_{gene2}'] = 'Background'
    combined_markers = np.select(
        [gene1_high & gene2_high, gene1_high, gene2_high],
        ['Autocrine', gene1, gene2],
        default='Background'
    )
    adata.obs.loc[marked_cells, f'{gene1}_{gene2}'] = combined_markers[marked_cells]
    
    highlight_palette = {
        gene1: 'blue', 
        gene2: 'red',  
        'Autocrine': 'purple',  
        'Background': 'lightgrey'  
    }
    
    size_params = {}
    if size_factor is not None:
        fig = sc.pl.embedding(adata, basis=embedding, return_fig=True)  
        default_sizes = fig.axes[0].collections[0].get_sizes()
        default_size = np.mean(default_sizes) if len(default_sizes) > 0 else 20
        plt.close(fig)
        
        adata.obs['point_size'] = default_size
        for category in [gene1, gene2, 'Autocrine']:
            adata.obs.loc[adata.obs[f'{gene1}_{gene2}'] == category, 'point_size'] *= size_factor
        size_params["size"] = adata.obs['point_size']
    
    sc.pl.embedding(
        adata, 
        basis=embedding, 
        color=[f'{gene1}_{gene2}'], 
        palette=highlight_palette, 
        title=f'{gene1}-{gene2}',
        return_fig=True,
        **size_params
    )
    
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {savefig}")
    
    if not saveflag:
        del adata.obs[f'{gene1}_{gene2}']
        if 'point_size' in adata.obs: 
            del adata.obs['point_size']
    
    plt.show()

def getPatternDistribution(W, phenotype_interval, labels=None, savefig=None, fontsize=12):
    """
    Plot violin plots to show the distribution of each feature in the W matrix across binary phenotype samples.

    Parameters:
    W (np.ndarray): Input matrix with shape (n_samples, n_features).
    labels (list or None): Category labels. Then this will be used as the x-axis label (corresponding to the key order of phenotype_interval)
    phenotype_interval (dict): Boundary points for different types of samples.
    savefig (str): Path to save the generated image, default is None (do not save).
    fontsize (int): Font size for titles and labels, default is 10.
    """
    num_features = W.shape[1] 
    num_subplots = int(np.ceil(np.sqrt(num_features)))
    num_rows = num_subplots
    num_cols = num_subplots if num_features % num_subplots == 0 else num_subplots + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    axes = axes.flatten()

    group_keys = list(phenotype_interval.keys())

    if labels is None:
        xtick_labels = [str(k) for k in group_keys]
    else:
        xtick_labels = labels

    for i in range(num_features):
        ax = axes[i]
        p = []
        for k in group_keys:
            start_idx, end_idx = phenotype_interval[k]
            p.append(W[:, i][start_idx:end_idx + 1])

        sns.violinplot(ax=ax, data=p, palette='Set2')

        ax.set_title(f"Pattern {i + 1}", fontsize=fontsize)
        ax.set_xticks(list(range(len(group_keys))))
        ax.set_xticklabels(xtick_labels, fontsize=fontsize)
        ax.set_ylabel("Values", fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {savefig}")

    plt.show()

def refine_beta(sample_id, pred, adj, method="mean", include_self=True):
    """
    Smooth continuous predictions using a sparse k-nearest-neighbor adjacency matrix.
    
    Parameters
    ----------
    sample_id : array-like, shape (n,)
        Unique identifier for each spot (e.g., barcode). Must be in the same order as the rows of adj.
    pred : array-like, shape (n,)
        Initial continuous prediction for each spot.
    adj : scipy.sparse.csr_matrix, shape (n, n)
        Sparse adjacency matrix (e.g., built by k-nearest-neighbor graph). Non-zero entries in row i indicate the neighbors of spot i.
    method : {"mean", "median"}, default "mean"
        Neighborhood aggregation method.
    include_self : bool, default True
        Whether to include the spot's own prediction when computing the neighborhood mean/median (original implementation includes self).
    
    Returns
    -------
    refined_pred : pandas.Series, index=sample_id
        Smoothed predictions.
    """
    if not isinstance(adj, csr_matrix):
        adj = csr_matrix(adj)

    sample_id = np.asarray(sample_id)
    pred = np.asarray(pred, dtype=float)
    n = pred.shape[0]
    assert adj.shape == (n, n), "adj dimension must be consistent with pred/sample_id."

    refined = np.empty(n, dtype=float)

    for i in range(n):
        row = adj.getrow(i)
        nbs_idx = row.indices

        if include_self:
            if i in nbs_idx:
                idx = nbs_idx
            else:
                idx = np.concatenate(([i], nbs_idx))
        else:
            idx = nbs_idx if len(nbs_idx) > 0 else np.array([i])

        vals = pred[idx]
        if method == "mean":
            refined[i] = vals.mean()
        elif method == "median":
            refined[i] = np.median(vals)
        else:
            raise ValueError("Method not recognized. Use 'mean' or 'median'.")

    return pd.Series(refined, index=sample_id, name="refined_pred")

def plot_positive_fraction_per_celltype(
    adata,
    phe_col: str = "phe",
    celltype_col: str = "celltype",
    positive_value = '1',
    min_cells: int = 10,
    figsize: Tuple[float, float] = (6,4)
):


    obs = adata.obs.copy()

    total_counts = obs.groupby(celltype_col).size()

    positive_counts = obs[obs[phe_col] == positive_value].groupby(celltype_col).size()

    stat_df = pd.DataFrame({
        "total_cells": total_counts,
        "positive_cells": positive_counts
    }).fillna(0)

    stat_df["positive_fraction"] = stat_df["positive_cells"] / stat_df["total_cells"]

    stat_df = stat_df[stat_df["total_cells"] >= min_cells]

    stat_df = stat_df.sort_values(by="positive_fraction", ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(stat_df.index.astype(str), stat_df["positive_fraction"].values)

    ax.set_ylabel(f"Fraction {phe_col}={positive_value}")
    ax.set_xlabel(celltype_col)
    ax.set_title(f"Positive fraction per cell type ({phe_col}={positive_value})")
    ax.set_ylim(0, stat_df["positive_fraction"].max() * 1.1)

    for i, v in enumerate(stat_df["positive_fraction"].values):
        ax.text(i, v, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticklabels(stat_df.index.astype(str), rotation=45, ha="right")

    plt.tight_layout()
    return fig, ax, stat_df

def plot_Spatiallr(
    adata: AnnData,
    lr_pair: tuple[str, str],
    layer: str = None,
    topn_frac: float = 0.2,
    knn: int = 8,
    pt_size: float = 2.0,
    alpha_min: float = 0.1,
    max_cut: float = 0.95,
    figsize: tuple = (12, 6),
    dual_plot: bool = True
) -> plt.Figure:
    """
    Visualize the spatial co-localization of a ligand-receptor pair.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing spatial transcriptomics data.
    lr_pair : tuple[str, str]
        Ligand-receptor pair names, e.g., ('Ptn', 'Ptprz1').
    layer : str, optional
        Use a specific matrix from adata.layers; default is adata.X.
    topn_frac : float, optional
        Proportion of cells considered as high expression (default: 0.2).
    knn : int, optional
        Number of nearest neighbors (default: 8).
    pt_size : float, optional
        Size of scatter points (default: 2).
    alpha_min : float, optional
        Minimum transparency (default: 0.1).
    max_cut : float, optional
        Maximum cutoff for LR activity (default: 0.95).
    figsize : tuple, optional
        Size of the figure (default: (12, 6)).
    dual_plot : bool, optional
        Whether to show both expression and activity plots (default: True).

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the visualization results.
    """
    if lr_pair[0] not in adata.var_names or lr_pair[1] not in adata.var_names:
        raise ValueError("Ligand or receptor not found in gene names")
    
    expr = adata.layers[layer] if layer else adata.X
    if issparse(expr):
        expr = expr.toarray()  
    
    if 'spatial' not in adata.obsm:
        raise KeyError("Spatial coordinates not found in adata.obsm['spatial']")
    location = adata.obsm['spatial']
    
    adata2 = adata[adata.obs['refined_beta'] > 0] 
    expr2 = expr[adata.obs['refined_beta'] > 0]
    location2 = location[adata.obs['refined_beta'] > 0]
    
    nn_model = NearestNeighbors(n_neighbors=knn+1).fit(location2)
    _, nn_indices = nn_model.kneighbors(location2)
    
    lig_idx = adata.var_names.get_loc(lr_pair[0])
    rec_idx = adata.var_names.get_loc(lr_pair[1])
    ligand = expr2[:, lig_idx]
    receptor = expr2[:, rec_idx]
    
    neighbor_expr = np.zeros((2, expr2.shape[0]))
    for i in range(expr2.shape[0]):
        neighbors = nn_indices[i, 1:]
        neighbor_expr[0, i] = np.max(ligand[neighbors])
        neighbor_expr[1, i] = np.max(receptor[neighbors])
    
    lr_activity = np.maximum(ligand * neighbor_expr[1], receptor * neighbor_expr[0])
    lr_cut = np.quantile(lr_activity, max_cut)
    lr_activity = np.clip(lr_activity, None, lr_cut)
    
    n_cells = expr2.shape[0]
    topn = int(topn_frac * n_cells)
    
    lig_order = np.argsort(-ligand + np.random.randn(n_cells) * 1e-6)
    lig_high = lig_order[:topn] if np.sum(ligand > 0) >= topn else np.where(ligand > 0)[0]
    rec_order = np.argsort(-receptor + np.random.randn(n_cells) * 1e-6)
    rec_high = rec_order[:topn] if np.sum(receptor > 0) >= topn else np.where(receptor > 0)[0]
    
    exp_type = np.zeros(n_cells, dtype=int)
    exp_type[lig_high] = 1
    exp_type[rec_high] = 2
    exp_type[np.intersect1d(lig_high, rec_high)] = 3
    
    plot_df = pd.DataFrame({
        'x': location2[:, 0],
        'y': location2[:, 1],
        'type': exp_type,
        'activity': lr_activity
    })
    
    single_width = figsize[0] / 2
    if dual_plot:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2]) 
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(single_width*1.2, figsize[1]))
    
    if dual_plot:
        ax1.scatter(location[:, 0], location[:, 1], color='gray', alpha=0.2, s=pt_size)
        colors = ['gray', 'red', 'green', 'blue']
        labels = ['Both low', 'Ligand high', 'Receptor high', 'Both high']
        for i in range(4):
            mask = plot_df['type'] == i
            ax1.scatter(
                plot_df.loc[mask, 'x'], 
                plot_df.loc[mask, 'y'], 
                c=colors[i], 
                label=labels[i],
                s=pt_size
            )
        ax1.legend(loc='best')
        ax1.set_title(f"{lr_pair[0]}-{lr_pair[1]} Expression")
        ax1.invert_yaxis()
    
    alpha = (lr_activity - lr_activity.min()) / (lr_activity.max() - lr_activity.min()) * (1 - alpha_min) + alpha_min
    ax2.scatter(location[:, 0], location[:, 1], color='gray', alpha=0.1, s=pt_size)
    scatter = ax2.scatter(
        plot_df['x'], 
        plot_df['y'], 
        c=plot_df['activity'], 
        cmap='RdGy_r',
        s=pt_size,
        alpha=alpha
    )
    plt.colorbar(scatter, ax=ax2, label='LR Activity', shrink=0.8) 
    ax2.set_title(f"{lr_pair[0]}-{lr_pair[1]} Activity")
    ax2.invert_yaxis()

    for ax in [ax1, ax2] if dual_plot else [ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.tight_layout()
    return fig

def _de_select_genes_for_cluster(
    counts_gxc: pd.DataFrame,           # genes x cells (float32)
    meta: pd.DataFrame,                 # index=cell, col 'cell_type'
    cluster: str,
    method: Literal['wilcoxon','perm_logfc'] = 'wilcoxon',
    alpha: float = 0.05,
    logfc_min: float = 0.0,
    top_n: Optional[int] = 500,
    perm_iter: int = 200,
    random_state: int = 0,
) -> set:

    rng = np.random.default_rng(random_state)
    cells_in = meta.index[meta['cell_type'] == cluster].to_numpy()
    cells_out = meta.index[meta['cell_type'] != cluster].to_numpy()

    if len(cells_in) == 0 or len(cells_out) == 0:
        return set()

    mu_in = counts_gxc[cells_in].mean(axis=1).astype('float64')
    mu_out = counts_gxc[cells_out].mean(axis=1).astype('float64')
    logfc = (np.log1p(mu_in) - np.log1p(mu_out)).values
    genes = counts_gxc.index.to_numpy()

    if method == 'wilcoxon':
        pvals = np.ones(len(genes), dtype='float64')
        for gi, g in enumerate(genes):
            x = counts_gxc.loc[g, cells_in].values
            y = counts_gxc.loc[g, cells_out].values
            try:
                stat = mannwhitneyu(x, y, alternative='greater', method='asymptotic')
                pvals[gi] = stat.pvalue
            except Exception:
                pvals[gi] = 1.0
    else:  # perm_logfc
        obs = (mu_in.values - mu_out.values)  
        count_ge = np.zeros_like(obs, dtype='int32')
        labels = meta['cell_type'].to_numpy()
        in_mask = (labels == cluster)

        for _ in range(perm_iter):
            shuf = in_mask.copy()
            rng.shuffle(shuf)
            mu_in_b = counts_gxc.loc[:, meta.index[shuf]].mean(axis=1).values
            mu_out_b = counts_gxc.loc[:, meta.index[~shuf]].mean(axis=1).values
            stat_b = mu_in_b - mu_out_b
            count_ge += (stat_b >= obs)

        pvals = (count_ge + 1.0) / (perm_iter + 1.0)  

    keep = (pvals <= alpha) & (logfc >= logfc_min)
    idx = np.where(keep)[0]
    if idx.size == 0:
        return set()
    if top_n is not None:
        sel = idx[np.argsort(-logfc[idx])[:top_n]]
    else:
        sel = idx
    return set(genes[sel])

def CCIdetect(
    adata,
    celltype_key: str,
    interactions: pd.DataFrame,
    senders: Optional[List[str]] = None,
    receivers: Optional[List[str]] = None,
    iterations: int = 1000,
    threshold: float = 0.1,
    pvalue_threshold: float = 0.05,
    subsampling_fraction: float = 1.0,
    de_enable: bool = True,
    de_method: Literal['wilcoxon','perm_logfc'] = 'wilcoxon',
    de_alpha: float = 0.05,
    de_logfc_min: float = 0.0,
    de_top_n: Optional[int] = 1000,
    de_perm_iter: int = 200,
    de_random_state: int = 0,
    fdr_method: Literal['none','bh','by'] = 'none',
    perm_random_state: Optional[int] = 0,   
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:

    logger = type('L', (), {'info': print})()
    logger.info(f"Running CCIdetect(optimized): iterations={iterations}, threshold={threshold}, "
                f"pvalue_threshold={pvalue_threshold}, senders={senders}, receivers={receivers}, "
                f"subsampling_fraction={subsampling_fraction}, de_enable={de_enable}, "
                f"de_method={de_method}, fdr_method={fdr_method}, perm_random_state={perm_random_state}")

    if not 0 < subsampling_fraction <= 1:
        raise ValueError("subsampling_fraction must be between 0 and 1")

    meta = pd.DataFrame({
        'cell': adata.obs.index,
        'cell_type': adata.obs[celltype_key].astype(str)
    })

    """
    X = adata.X.T
    if hasattr(X, 'toarray'):
        X = X.toarray()
    counts = pd.DataFrame(
        X,
        index=adata.var.index,      # genes
        columns=adata.obs.index     # cells
    ).astype('float32')
    """
    counts = pd.DataFrame(
        adata.X.T.toarray() if hasattr(adata.X.T, 'toarray') else adata.X.T,
        index=adata.var.index,
        columns=adata.obs.index
    ).astype('float32')

    common_cells = sorted(set(meta['cell']).intersection(counts.columns))
    meta = meta[meta['cell'].isin(common_cells)].set_index('cell')
    counts = counts[common_cells]

    all_cluster_names = sorted(meta['cell_type'].unique())
    senders = senders if senders is not None else all_cluster_names
    receivers = receivers if receivers is not None else all_cluster_names

    invalid_senders = set(senders) - set(all_cluster_names)
    invalid_receivers = set(receivers) - set(all_cluster_names)
    if invalid_senders:
        raise ValueError(f"Invalid sender cell types: {invalid_senders}")
    if invalid_receivers:
        raise ValueError(f"Invalid receiver cell types: {invalid_receivers}")

    meta = meta[meta['cell_type'].isin(senders + receivers)]
    counts = counts[meta.index]  

    if subsampling_fraction < 1.0:
        sampled_cells = meta.sample(frac=subsampling_fraction, random_state=42).index
        meta = meta.loc[sampled_cells]
        counts = counts[sampled_cells]
        logger.info(f"Subsampled to {len(sampled_cells)} cells (fraction={subsampling_fraction})")

    cluster_names = sorted(set(senders).union(receivers))

    interactions = interactions.copy()
    valid_genes = set(counts.index)
    interactions_valid = interactions[
        interactions['ligand'].isin(valid_genes) &
        interactions['receptor'].isin(valid_genes)
    ].copy()
    if interactions_valid.empty:
        raise ValueError("No valid ligand-receptor pairs found in counts data.")

    lr_genes_for_de = sorted(set(interactions_valid['ligand']) |
                             set(interactions_valid['receptor']))
    lr_genes_for_de = [g for g in lr_genes_for_de if g in counts.index]

    sender_up_map = {}
    receiver_up_map = {}
    sender_up_union = set()
    receiver_up_union = set()

    if de_enable:
        counts_gxc = counts.loc[lr_genes_for_de]  # genes x cells

        for c in senders:
            up = _de_select_genes_for_cluster(
                counts_gxc, meta, c,
                method=de_method,
                alpha=de_alpha,
                logfc_min=de_logfc_min,
                top_n=de_top_n,
                perm_iter=de_perm_iter,
                random_state=de_random_state,
            )
            sender_up_map[c] = up
            sender_up_union |= up

        for c in receivers:
            up = _de_select_genes_for_cluster(
                counts_gxc, meta, c,
                method=de_method,
                alpha=de_alpha,
                logfc_min=de_logfc_min,
                top_n=de_top_n,
                perm_iter=de_perm_iter,
                random_state=de_random_state,
            )
            receiver_up_map[c] = up
            receiver_up_union |= up

        interactions_de = interactions_valid[
            interactions_valid['ligand'].isin(sender_up_union) &
            interactions_valid['receptor'].isin(receiver_up_union)
        ].copy()
        logger.info(f"DE shrink (LR space): {len(interactions_valid)} -> {len(interactions_de)} LR pairs")

        if interactions_de.empty:
            logger.info("DE too stringent; falling back to unfiltered LR interactions "
                        "(in LR gene space).")
            interactions = interactions_valid.copy()
            lr_gene_set = set(lr_genes_for_de)
            sender_up_map = {c: lr_gene_set for c in senders}
            receiver_up_map = {c: lr_gene_set for c in receivers}
        else:
            interactions = interactions_de
    else:
        interactions = interactions_valid.copy()
        lr_gene_set = set(lr_genes_for_de)
        sender_up_map = {c: lr_gene_set for c in senders}
        receiver_up_map = {c: lr_gene_set for c in receivers}

    lr_genes_final = sorted(set(interactions['ligand']) |
                            set(interactions['receptor']))
    counts = counts.loc[lr_genes_final]  

    interactions = interactions.copy()
    interactions['interaction'] = interactions['ligand'] + '_' + interactions['receptor']
    interactions.set_index('interaction', inplace=True)

    ligands = interactions['ligand'].values
    receptors = interactions['receptor'].values

    cluster_pairs = [(s, r) for s in senders for r in receivers]
    pair_columns = [f"{s}_{r}" for s, r in cluster_pairs]

    counts_mat = counts.values
    genes_index = counts.index
    G, N = counts_mat.shape

    labels = meta.loc[counts.columns, 'cell_type'].to_numpy()

    clusters_means_mat = np.zeros((G, len(cluster_names)), dtype='float32')
    for j, cluster in enumerate(cluster_names):
        mask = (labels == cluster)
        if not np.any(mask):
            continue
        clusters_means_mat[:, j] = counts_mat[:, mask].mean(axis=1)

    clusters_means = pd.DataFrame(
        clusters_means_mat, index=genes_index, columns=cluster_names
    )

    lig_idx = genes_index.get_indexer(ligands)
    rec_idx = genes_index.get_indexer(receptors)

    K = len(interactions)
    P = len(cluster_pairs)

    ligand_means = clusters_means.values[lig_idx, :]   # (K x C)
    receptor_means = clusters_means.values[rec_idx, :] # (K x C)

    means = np.zeros((K, P), dtype='float32')
    allow_mask = np.zeros((K, P), dtype=bool)

    for j, (s, r) in enumerate(cluster_pairs):
        s_idx = cluster_names.index(s)
        r_idx = cluster_names.index(r)

        lig_col = ligand_means[:, s_idx]
        rec_col = receptor_means[:, r_idx]
        col = np.where(
            (lig_col == 0) | (rec_col == 0),
            0.0,
            np.sqrt(lig_col * rec_col)
        )

        allow = (np.isin(ligands, list(sender_up_map.get(s, set()))) &
                 np.isin(receptors, list(receiver_up_map.get(r, set()))))
        allow_mask[:, j] = allow
        col[~allow] = 0.0
        means[:, j] = col

    means_df = pd.DataFrame(means, index=interactions.index, columns=pair_columns)

    percents_mat = np.zeros((G, len(cluster_names)), dtype='float32')
    for j, cluster in enumerate(cluster_names):
        mask = (labels == cluster)
        if not np.any(mask):
            continue
        # (counts_mat[:, mask] > 0) -> bool, mean(axis=1)
        percents_mat[:, j] = (counts_mat[:, mask] > 0).mean(axis=1)

    percents = pd.DataFrame(
        percents_mat, index=genes_index, columns=cluster_names
    )

    ligand_perc = percents.values[lig_idx, :]   # (K x C)
    receptor_perc = percents.values[rec_idx, :] # (K x C)

    s_idx_vec = np.array([cluster_names.index(s) for s, _ in cluster_pairs], dtype=int)
    r_idx_vec = np.array([cluster_names.index(r) for _, r in cluster_pairs], dtype=int)

    lig_pos_pairs = ligand_perc[:, s_idx_vec]   # (K x P)
    rec_pos_pairs = receptor_perc[:, r_idx_vec] # (K x P)

    valid_mask = (means > 0) & (lig_pos_pairs >= threshold) & (rec_pos_pairs >= threshold)
    compare_mask = valid_mask & allow_mask

    if perm_random_state is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(perm_random_state)

    count_ge = np.zeros((K, P), dtype=np.int32)

    for _ in tqdm(range(iterations), desc="CCI permutations"):
        shuffled_labels = rng.permutation(labels)

        shuffled_means_mat = np.zeros((G, len(cluster_names)), dtype='float32')
        for j, cluster in enumerate(cluster_names):
            mask = (shuffled_labels == cluster)
            if not np.any(mask):
                continue
            shuffled_means_mat[:, j] = counts_mat[:, mask].mean(axis=1)

        lm_b = shuffled_means_mat[lig_idx, :]  # (K x C)
        rm_b = shuffled_means_mat[rec_idx, :]  # (K x C)

        shuf = np.sqrt(lm_b[:, s_idx_vec] * rm_b[:, r_idx_vec])  # (K x P)
        shuf[(lm_b[:, s_idx_vec] == 0) | (rm_b[:, r_idx_vec] == 0)] = 0.0

        count_ge += ((shuf >= means) & compare_mask).astype(np.int32)

    pvals = (count_ge + 1.0) / (iterations + 1.0)
    pvals[~allow_mask] = 1.0
    pvals[~valid_mask] = 1.0

    pvalues_df = pd.DataFrame(pvals, index=interactions.index, columns=pair_columns)

    try:
        from statsmodels.stats.multitest import multipletests
        has_statsmodels = True
    except Exception:
        has_statsmodels = False

    pvalues_adj_df = None
    if fdr_method != 'none':
        if not has_statsmodels:
            print("[WARNING] statsmodels not installed; FDR correction skipped, "
                  "returning raw p-values.")
        else:
            method = 'fdr_bh' if fdr_method == 'bh' else 'fdr_by'
            flat = pvalues_df.values.ravel()
            _, p_adj_flat, _, _ = multipletests(flat, method=method)
            p_adj = p_adj_flat.reshape(pvalues_df.shape)
            pvalues_adj_df = pd.DataFrame(
                p_adj, index=pvalues_df.index, columns=pvalues_df.columns
            )

    judge = pvalues_adj_df if pvalues_adj_df is not None else pvalues_df

    significant_means = means_df.copy()
    significant_means[judge > pvalue_threshold] = np.nan

    interactions_data = interactions[['ligand', 'receptor']].reset_index()

    pvalues_out = pd.concat(
        [interactions_data, pvalues_df.reset_index(drop=True)], axis=1
    )
    means_out = pd.concat(
        [interactions_data, means_df.reset_index(drop=True)], axis=1
    )
    significant_means_out = pd.concat(
        [interactions_data, significant_means.reset_index(drop=True)], axis=1
    )

    pvalues_adj_out = None
    if pvalues_adj_df is not None:
        pvalues_adj_out = pd.concat(
            [interactions_data, pvalues_adj_df.reset_index(drop=True)], axis=1
        )

    return pvalues_out, means_out, significant_means_out, pvalues_adj_out

def cpdb_exact_target(means,target_cells):
    import re
    
    t_dict=[]
    for t in target_cells:
        escaped_str = re.escape('_'+t)
        target_names=means.columns[means.columns.str.contains(escaped_str)].tolist()
        t_dict+=target_names
    target_sub=means[means.columns[:3].tolist()+t_dict]
    return target_sub

def cpdb_exact_source(means,source_cells):
    import re
    
    t_dict=[]
    for t in source_cells:
        escaped_str = re.escape(t+'_')
        source_names=means.columns[means.columns.str.contains(escaped_str)].tolist()
        t_dict+=source_names
    source_sub=means[means.columns[:3].tolist()+t_dict]
    return source_sub

def cci_interacting_heatmap(adata, 
                             celltype_key,
                             means,
                             pvalues,
                             source_cells,
                             target_cells,
                             min_means=3,  # Only LR pairs with total expression > min_means across the involved cell pairs are displayed.
                             nodecolor_dict=None,
                             ax=None,
                             figsize=(2,6),
                             fontsize=12,
                             return_table=False,
                             pval_threshold=None 
                             ):

    if nodecolor_dict is not None:
        type_color_all = nodecolor_dict
    else:
        if f'{celltype_key}_colors' in adata.uns:
            type_color_all = dict(zip(adata.obs[celltype_key].cat.categories, adata.uns[f'{celltype_key}_colors']))
        else:
            if len(adata.obs[celltype_key].cat.categories) > 28:
                type_color_all = dict(zip(adata.obs[celltype_key].cat.categories, sc.pl.palettes.default_102))
            else:
                type_color_all = dict(zip(adata.obs[celltype_key].cat.categories, sc.pl.palettes.zeileis_28))

    sub_means = cpdb_exact_target(means, target_cells)
    sub_means = cpdb_exact_source(sub_means, source_cells)

    sub_means = sub_means.loc[~sub_means['ligand'].isnull()]
    sub_means = sub_means.loc[~sub_means['receptor'].isnull()]

    new = sub_means.iloc[:, 3:]
    new.index = sub_means['interaction'].tolist()
    cor = new.loc[new.sum(axis=1)[new.sum(axis=1) > min_means].index]

    sub_p = pvalues.set_index('interaction').loc[cor.index, cor.columns]

    corr_mat = cor.stack().reset_index(name="means")           
    sub_p_mat = sub_p.stack().reset_index(name="pvalue")
    corr_mat = corr_mat.merge(sub_p_mat, on=['level_0', 'level_1'], how='left')

    eps = 1e-3
    corr_mat['pvalue'] = corr_mat['pvalue'].astype(float)
    corr_mat['-logp'] = -np.log10(corr_mat['pvalue'].fillna(1.0) + eps)


    if pval_threshold is not None:
        corr_mat = corr_mat[corr_mat['pvalue'] < float(pval_threshold)]
        if corr_mat.empty:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
                ax.axis('off')
                ax.text(0.5, 0.5, f'No points with p < {pval_threshold}', ha='center', va='center', fontsize=fontsize)
                return ax if not return_table else cor.iloc[0:0]
            else:
                ax.text(0.5, 0.5, f'No points with p < {pval_threshold}', ha='center', va='center', fontsize=fontsize, transform=ax.transAxes)
                return ax if not return_table else cor.iloc[0:0]


    df_col = corr_mat['level_1'].drop_duplicates().to_frame()
    df_col['Source'] = df_col.level_1.apply(lambda x: x.split('_')[0])
    df_col['Target'] = df_col.level_1.apply(lambda x: x.split('_')[1])
    df_col.set_index('level_1', inplace=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax

    col_ha = HeatmapAnnotation(
        Source=anno_simple(
            df_col.Source,
            colors=[type_color_all[i] for i in df_col.Source.unique()],
            text_kws={'color': 'black', 'rotation': 0, 'fontsize': fontsize},
            legend=True, add_text=False
        ),
        Target=anno_simple(
            df_col.Target,
            colors=[type_color_all[i] for i in df_col.Target.unique()],
            text_kws={'color': 'black', 'rotation': 0, 'fontsize': fontsize},
            legend=True, add_text=False
        ),
        verbose=0, label_side='left',
        label_kws={'horizontalalignment': 'right', 'fontsize': fontsize}
    )
    cm = DotClustermapPlotter(
        corr_mat, x='level_1', y='level_0', value='means',
        c='means', s='-logp', cmap='Reds', vmin=0,
        top_annotation=col_ha,
        row_dendrogram=True,
        row_cluster_metric='euclidean', 
        col_cluster_metric='euclidean', 
        show_rownames=True, show_colnames=True
    )

    cm.ax_heatmap.grid(which='minor', color='gray', linestyle='--')

    for ax_i in plt.gcf().axes:
        if hasattr(ax_i, 'get_ylabel') and ax_i.get_ylabel() == 'means':
            cbar = ax_i
            cbar.tick_params(labelsize=fontsize)
            cbar.set_ylabel('means', fontsize=fontsize)
        ax_i.grid(False)

    ax_list = plt.gcf().axes
    if len(ax_list) > 6:
        ax_list[6].set_xticklabels(ax_list[6].get_xticklabels(), fontsize=fontsize)
        ax_list[6].set_yticklabels(ax_list[6].get_yticklabels(), fontsize=fontsize)

    if return_table:
        return cor
    else:
        return ax


def extract_interaction_edges(
    pvals: pd.DataFrame,
    alpha: float = 0.05,
    default_sep: str = "_",  # Cell type sep
    symmetrical: bool = True,
) -> pd.DataFrame:
    """exacting the significant Cell-cell interaction table: interaction_edges"""
    
    all_intr = pvals.rename(columns={"interaction": "interacting_pair"}).copy()
    intr_pairs = all_intr["interacting_pair"]
    
    col_start = 3  # interactive data starts at column 4

    all_int = all_intr.iloc[:, col_start:].T
    all_int.columns = intr_pairs

    cell_types = sorted(
        list(set([y for z in [x.split(default_sep) for x in all_intr.columns[col_start:]] for y in z]))
    )

    cell_types_comb = ["_".join(x) for x in product(cell_types, cell_types)]
    
    cell_types_keep = [ct for ct in all_int.index if ct in cell_types_comb]
    all_int = all_int.loc[cell_types_keep]

    all_count = all_int.melt(ignore_index=False).reset_index()
    
    all_count["significant"] = all_count.value < alpha
    
    count1x = all_count[["index", "significant"]].groupby("index").agg({"significant": "sum"})
    tmp = pd.DataFrame([x.split("_") for x in count1x.index])
    count_final = pd.concat([tmp, count1x.reset_index(drop=True)], axis=1)
    count_final.columns = ["SOURCE", "TARGET", "COUNT"]

    return count_final

def cci_heatmap(adata: anndata.AnnData, interaction_edges: pd.DataFrame,
                 celltype_key: str, nodecolor_dict=None, ax=None,
                 source_cells=None, target_cells=None,
                 figsize=(3, 3), fontsize=11, rotate=False, legend=True,
                 legend_kws={'fontsize': 8, 'bbox_to_anchor': (5, -0.5), 'loc': 'center left'},
                 return_table=False, **kwargs):
    
    # Color dictionary setup
    if nodecolor_dict is not None:
        type_color_all = nodecolor_dict
    else:
        if f'{celltype_key}_colors' in adata.uns:
            type_color_all = dict(zip(adata.obs[celltype_key].cat.categories, adata.uns[f'{celltype_key}_colors']))
        else:
            if len(adata.obs[celltype_key].cat.categories) > 28:
                type_color_all = dict(zip(adata.obs[celltype_key].cat.categories, sc.pl.palettes.default_102))
            else:
                type_color_all = dict(zip(adata.obs[celltype_key].cat.categories, sc.pl.palettes.zeileis_28))

    # Filter interaction edges
    corr_mat = interaction_edges.copy()
    if source_cells is not None and target_cells is None:
        corr_mat = corr_mat.loc[corr_mat['SOURCE'].isin(source_cells)]
    elif source_cells is None and target_cells is not None:
        corr_mat = corr_mat.loc[corr_mat['TARGET'].isin(target_cells)]
    elif source_cells is not None and target_cells is not None:
        corr_mat = corr_mat.loc[corr_mat['TARGET'].isin(source_cells)]
        corr_mat = corr_mat.loc[corr_mat['SOURCE'].isin(target_cells)]

    # Prepare row and column dataframes
    df_row = corr_mat['SOURCE'].drop_duplicates().to_frame()
    df_row['Celltype'] = df_row['SOURCE']
    df_row.set_index('SOURCE', inplace=True)

    df_col = corr_mat['TARGET'].drop_duplicates().to_frame()
    df_col['Celltype'] = df_col['TARGET']
    df_col.set_index('TARGET', inplace=True)

    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax

    # Row and column annotations
    if not rotate:
        row_ha = HeatmapAnnotation(
            TARGET=anno_simple(
                df_row.Celltype,
                colors=[type_color_all[i] for i in df_row.Celltype],
                add_text=False,
                text_kws={'color': 'black', 'rotation': 0, 'fontsize': 10},
                legend=False
            ),
            legend_gap=7,
            axis=1,
            verbose=0,
            label_kws={'rotation': 90, 'horizontalalignment': 'right', 'fontsize': 0}
        )
    else:
        row_ha = HeatmapAnnotation(
            TARGET=anno_simple(
                df_row.Celltype,
                colors=[type_color_all[i] for i in df_row.Celltype],
                add_text=False,
                text_kws={'color': 'black', 'rotation': 0, 'fontsize': 10},
                legend=False
            ),
            legend_gap=7,
            axis=0,
            verbose=0,
            label_kws={'rotation': 90, 'horizontalalignment': 'right', 'fontsize': 0}
        )

    if not rotate:
        col_ha = HeatmapAnnotation(
            SOURCE=anno_simple(
                df_col.Celltype,
                colors=[type_color_all[i] for i in df_col.Celltype],
                legend=False,
                add_text=False
            ),
            verbose=0,
            label_kws={'horizontalalignment': 'right', 'fontsize': 0},
            legend_kws={'ncols': 1},
            legend=False,
            legend_hpad=7,
            legend_vpad=5,
            axis=0
        )
    else:
        col_ha = HeatmapAnnotation(
            SOURCE=anno_simple(
                df_col.Celltype,
                colors=[type_color_all[i] for i in df_col.Celltype],
                legend=False,
                add_text=False
            ),
            verbose=0,
            label_kws={'horizontalalignment': 'right', 'fontsize': 0},
            legend_kws={'ncols': 1},
            legend=False,
            legend_hpad=7,
            legend_vpad=5,
            axis=1
        )

    import PyComplexHeatmap as pch
    if pch.__version__ > '1.7':
        hue_arg = None
    else:
        hue_arg = 'SOURCE'

    # DotClustermapPlotter
    if rotate:
        cm = DotClustermapPlotter(
            corr_mat,
            y='SOURCE',
            x='TARGET',
            value='COUNT',
            hue=hue_arg,
            legend_gap=7,
            top_annotation=col_ha,
            left_annotation=row_ha,
            c='COUNT',
            s='COUNT',
            cmap='Reds',
            vmin=0,
            show_rownames=False,
            show_colnames=False,
            row_dendrogram=False,
            col_names_side='left',
            legend=legend,
            row_cluster_metric='euclidean',
            col_cluster_metric='euclidean',
            **kwargs
        )
    else:
        cm = DotClustermapPlotter(
            corr_mat,
            x='SOURCE',
            y='TARGET',
            value='COUNT',
            hue=hue_arg,
            legend_gap=7,
            top_annotation=row_ha,
            left_annotation=col_ha,
            c='COUNT',
            s='COUNT',
            cmap='Reds',
            vmin=0,
            show_rownames=False,
            show_colnames=False,
            row_dendrogram=False,
            col_names_side='top',
            legend=legend,
            row_cluster_metric='euclidean',
            col_cluster_metric='euclidean',
            **kwargs
        )

    cm.ax_heatmap.grid(which='minor', color='gray', linestyle='--', alpha=0.5)
    cm.ax_heatmap.grid(which='major', color='black', linestyle='-', linewidth=0.5)
    cm.cmap_legend_kws = {'ncols': 1}

    # Adjust axes labels
    if not rotate:
        for ax in plt.gcf().axes:
            if hasattr(ax, 'get_ylabel'):
                if ax.get_ylabel() == 'COUNT':
                    cbar = ax
                    cbar.tick_params(labelsize=fontsize)
                    cbar.set_ylabel('COUNT', fontsize=fontsize)
                if ax.get_xlabel() == 'SOURCE':
                    ax.xaxis.set_label_position('top')
                    ax.set_ylabel('Target', fontsize=fontsize)
                if ax.get_ylabel() == 'TARGET':
                    ax.xaxis.set_label_position('top')
                    ax.set_xlabel('Source', fontsize=fontsize)
            ax.grid(False)
    else:
        for ax in plt.gcf().axes:
            if hasattr(ax, 'get_ylabel'):
                if ax.get_ylabel() == 'COUNT':
                    cbar = ax
                    cbar.tick_params(labelsize=fontsize)
                    cbar.set_ylabel('COUNT', fontsize=fontsize)
                if ax.get_ylabel() == 'SOURCE':
                    ax.xaxis.set_label_position('top')
                    ax.set_xlabel('Target', fontsize=fontsize)
                if ax.get_xlabel() == 'TARGET':
                    ax.xaxis.set_label_position('top')
                    ax.set_ylabel('Source', fontsize=fontsize)
            ax.grid(False)

    # Legend setup
    handles = [plt.Line2D([0], [0], color=type_color_all[cell], lw=4) for cell in type_color_all.keys()]
    labels = type_color_all.keys()

    # Place legend without causing layout issues
    if legend:
        plt.legend(handles, labels, 
                   borderaxespad=1, handletextpad=0.5, labelspacing=0.2, **legend_kws)

    plt.subplots_adjust(left=0.1, right=0.9, top=1.0, bottom=0.1)  # Adjust these values as needed

    if return_table:
        return corr_mat
    else:
        return ax

def cci_chord(adata:anndata.AnnData,interaction_edges:pd.DataFrame,
                      celltype_key:str,count_min=50,nodecolor_dict=None,
                      fontsize=12,padding=80,radius=100,save='chord.svg',
                      rotation=0,bg_color = "#ffffff",bg_transparancy = 1.0):
    import itertools
    import openchord as ocd
    data=interaction_edges.loc[interaction_edges['COUNT']>count_min].iloc[:,:2]
    data = list(itertools.chain.from_iterable((i, i[::-1]) for i in data.values))
    matrix = pd.pivot_table(
        pd.DataFrame(data), index=0, columns=1, aggfunc="size", fill_value=0
    ).values.tolist()
    unique_names = sorted(set(itertools.chain.from_iterable(data)))

    matrix_df = pd.DataFrame(matrix, index=unique_names, columns=unique_names)

    if nodecolor_dict!=None:
        type_color_all=nodecolor_dict
    else:
        if '{}_colors'.format(celltype_key) in adata.uns:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,adata.uns['{}_colors'.format(celltype_key)]))
        else:
            if len(adata.obs[celltype_key].cat.categories)>28:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.default_102))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.zeileis_28))
    
    fig=ocd.Chord(matrix, unique_names,radius=radius)
    fig.colormap=[type_color_all[u] for u in unique_names]
    fig.font_size=fontsize
    fig.padding = padding
    fig.rotation = rotation
    fig.bg_color = bg_color
    fig.bg_transparancy = bg_transparancy
    if save!=None:
        fig.save_svg(save)
    return fig

def cci_network(adata:anndata.AnnData,interaction_edges:pd.DataFrame,
                      celltype_key:str,nodecolor_dict=None,counts_min=50,
                       source_cells=None,target_cells=None,
                      edgeswidth_scale:int=1,nodesize_scale:int=1,
                      figsize:tuple=(4,4),title:str='',
                      fontsize:int=12,ax=None,
                     return_graph:bool=False):
    G=nx.DiGraph()
    for i in interaction_edges.index:
        if interaction_edges.loc[i,'COUNT']>counts_min:
            G.add_edge(interaction_edges.loc[i,'SOURCE'],
                       interaction_edges.loc[i,'TARGET'],
                       weight=interaction_edges.loc[i,'COUNT'],)
        else:
            G.add_edge(interaction_edges.loc[i,'SOURCE'],
                       interaction_edges.loc[i,'TARGET'],
                       weight=0,)

    if nodecolor_dict!=None:
        type_color_all=nodecolor_dict
    else:
        if '{}_colors'.format(celltype_key) in adata.uns:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,adata.uns['{}_colors'.format(celltype_key)]))
        else:
            if len(adata.obs[celltype_key].cat.categories)>28:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.default_102))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.zeileis_28))

    G_nodes_dict={}
    links = []
    for i in G.edges:
        if i[0] not in G_nodes_dict.keys():
            G_nodes_dict[i[0]]=0
        if i[1] not in G_nodes_dict.keys():
            G_nodes_dict[i[1]]=0
        links.append({"source": i[0], "target": i[1]})
        weight=G.get_edge_data(i[0],i[1])['weight']
        G_nodes_dict[i[0]]+=weight
        G_nodes_dict[i[1]]+=weight

    edge_li=[]
    for u,v in G.edges:
        if G.get_edge_data(u, v)['weight']>0:
            if source_cells==None and target_cells==None:
                edge_li.append((u,v))
            elif source_cells!=None and target_cells==None:
                if u in source_cells:
                    edge_li.append((u,v))
            elif source_cells==None and target_cells!=None:
                if v in target_cells:
                    edge_li.append((u,v))
            else:
                if u in source_cells and v in target_cells:
                    edge_li.append((u,v))


    import matplotlib.pyplot as plt
    import numpy as np
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize) 
    else:
        ax=ax
    pos = nx.circular_layout(G)
    p=dict(G.nodes)
    
    nodesize=np.array([G_nodes_dict[u] for u in G.nodes()])/nodesize_scale
    nodecolos=[type_color_all[u] for u in G.nodes()]
    nx.draw_networkx_nodes(G, pos, nodelist=p,node_size=nodesize,node_color=nodecolos)
    
    edgewidth = np.array([G.get_edge_data(u, v)['weight'] for u, v in edge_li])
    edgewidth=np.log10(edgewidth+1)/edgeswidth_scale
    edgecolos=[type_color_all[u] for u,o in edge_li]
    nx.draw_networkx_edges(G, pos,width=edgewidth,edge_color=edgecolos,edgelist=edge_li)
    plt.grid(False)
    plt.axis("off")
    
    pos1=dict()
    for i in G.nodes:
        pos1[i]=pos[i]
    from adjustText import adjust_text
    import adjustText
    from matplotlib import patheffects
    texts=[ax.text(pos1[i][0], 
               pos1[i][1],
               i,
               fontdict={'size':fontsize,'weight':'normal','color':'black'},
                path_effects=[patheffects.withStroke(linewidth=2, foreground='w')]
               ) for i in G.nodes if 'ENSG' not in i]
    if adjustText.__version__<='0.8':
        adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='red'),)
    else:
        adjust_text(texts,only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                    arrowprops=dict(arrowstyle='->', color='black'))
        
    plt.title(title,fontsize=fontsize+1)

    if return_graph==True:
        return G
    else:
        return ax

def cci_interacting_network(adata,
                             celltype_key,
                             means,
                             source_cells,
                             target_cells,
                             means_min=0,
                             means_sum_min=0,        
                             nodecolor_dict=None,
                             ax=None,
                             figsize=(6,6),
                             fontsize=10,
                             return_graph=False):
    """
    Creates and visualizes a network of cell-cell interactions.

    Parameters:
    adata : AnnData
        AnnData object containing cell type and associated data.
    celltype_key : str
        Column name for cell types.
    means : DataFrame
        DataFrame containing interaction strengths.
    source_cells : list
        List of source cell types.
    target_cells : list
        List of target cell types.
    means_sum_min : float, optional
        Minimum threshold for interaction strength of ligand-receptor (default is 0).
    means_min : float, optional
        Minimum threshold for the sum of individual interactions (default is 0).
    nodecolor_dict : dict, optional
        Dictionary mapping cell types to colors (default is None).
    ax : matplotlib.axes.Axes, optional
        Axes object for the plot (default is None).
    figsize : tuple, optional
        Size of the figure (default is (6, 6)).
    fontsize : int, optional
        Font size for node labels (default is 10).



    Returns:
    ax : matplotlib.axes.Axes
        Axes object with the drawn network.
    """
    from adjustText import adjust_text
    import re
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Determine node colors
    if nodecolor_dict:
        type_color_all = nodecolor_dict
    else:
        color_key = f"{celltype_key}_colors"
        categories = adata.obs[celltype_key].cat.categories
        if color_key in adata.uns:
            type_color_all = dict(zip(categories, adata.uns[color_key]))
        else:
            palette = sc.pl.palettes.default_102 if len(categories) > 28 else sc.pl.palettes.zeileis_28
            type_color_all = dict(zip(categories, palette))

    # Create a directed graph
    G = nx.DiGraph()

    # Filter the means DataFrame
    sub_means = cpdb_exact_target(means, target_cells)
    sub_means = cpdb_exact_source(sub_means, source_cells)
    sub_means = sub_means.loc[~sub_means['ligand'].isnull()]
    sub_means = sub_means.loc[~sub_means['receptor'].isnull()]

    # Build the graph
    nx_dict = {}
    for source_cell in source_cells:
        for target_cell in target_cells:
            key = f"{source_cell}_{target_cell}"
            nx_dict[key] = []
            escaped_str = re.escape(key)
            receptor_names = sub_means.columns[sub_means.columns.str.contains(escaped_str)].tolist()
            receptor_sub = sub_means[sub_means.columns[:3].tolist() + receptor_names]

            for j in receptor_sub.index:
                if receptor_sub.loc[j, receptor_names].sum() > means_sum_min:
                    for rece in receptor_names:
                        if receptor_sub.loc[j, rece] > means_min:
                            nx_dict[key].append(receptor_sub.loc[j, 'receptor'])
                            G.add_edge(source_cell, f'L:{receptor_sub.loc[j, "ligand"]}')
                            G.add_edge(f'L:{receptor_sub.loc[j, "ligand"]}', f'R:{receptor_sub.loc[j, "receptor"]}')
                            G.add_edge(f'R:{receptor_sub.loc[j, "receptor"]}', rece.split('_')[1])
            nx_dict[key] = list(set(nx_dict[key]))

    # means_sum_min:
    # If the sum of interaction strengths of a given ligand-receptor pair 
    # across all source-target cell pairs is greater than this threshold, 
    # then this ligand-receptor pair is considered for inclusion in the network.
    # (i.e., filter out globally weak ligand-receptor pairs)
    
    # means_min:
    # For a ligand-receptor pair that passes the global threshold (means_min),
    # only those specific cell-cell interactions (CCI) where the interaction 
    # strength exceeds this threshold will be shown in the network.
    # (i.e., filter out weak CCIs even if the ligand-receptor pair is overall strong)



    # Set colors for ligand and receptor nodes
    color_dict = type_color_all
    color_dict['ligand'] = '#a51616'  # Red for ligands
    color_dict['receptor'] = '#c2c2c2'  # Gray for receptors

    # Assign colors to nodes
    node_colors = [
        color_dict.get(node, 
                       color_dict['ligand'] if 'L:' in node 
                       else color_dict['receptor'])
        for node in G.nodes()
    ]

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Define shells for shell_layout
    source_nodes = [n for n in G.nodes() if n in source_cells]
    ligand_nodes = [n for n in G.nodes() if 'L:' in n]
    receptor_nodes = [n for n in G.nodes() if 'R:' in n]
    target_nodes = [n for n in G.nodes() if n in target_cells and n not in source_cells]
    shells = [source_nodes, ligand_nodes, receptor_nodes, target_nodes]

    # Use shell_layout
    pos = nx.shell_layout(G, nlist=shells)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#c2c2c2', ax=ax)

    # Add labels to the nodes
    texts = [
        ax.text(pos[node][0], pos[node][1], node,
                fontdict={'size': fontsize, 'weight': 'bold', 'color': 'black'})
        for node in G.nodes() if 'ENSG' not in node
    ]
    adjust_text(texts, only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                arrowprops=dict(arrowstyle='-', color='black'))

    # Remove axes
    ax.axis("off")

    if return_graph:
        return G
    else:
        return ax

def add_sender_from_CCIdetect_wide(refined_pathways, p_values, p_threshold=0.05):

    p_long = p_values.melt(
        id_vars=['interaction', 'ligand', 'receptor'],
        var_name='pair', value_name='p_value'
    )

    p_long[['sender', 'receiver']] = p_long['pair'].str.split('_', n=1, expand=True)

    sig_pairs = p_long[p_long['p_value'] < p_threshold].copy()

    sig_pairs = sig_pairs[['ligand', 'receptor', 'sender', 'receiver']].drop_duplicates()

    merged = refined_pathways.merge(
        sig_pairs,
        on=['ligand', 'receptor', 'receiver'],
        how='left'
    )
    """
    merged = (
        merged.groupby(['ligand', 'receptor', 'TF', 'loading', 'receiver'], as_index=False)
        .agg({'sender': lambda x: ','.join(sorted(set(x.dropna()))) if x.notna().any() else None})
    )
    """
    merged = merged.dropna(subset=['sender'])

    return merged

def refine_signal_network(
    adata,
    lig_rec: pd.DataFrame,
    cci_pvalues: pd.DataFrame = None,
    rec_tf_path: str = None,
    tf_target_path: str = None,
    expr_prop_thresh: float = 0.05,
    mean_expr_diff_thresh: float = 0.1,
    tau_thresh: float = 0,
    pval_thresh: float = 0.05,
    cell_type_field: str = "cell_type",
    TF_p_adj: bool = False,
    rec_p_adj: bool = False,
    mode: str = 'pre'
):

    if rec_tf_path is None:
        with importlib.resources.path("PopCILA.data", "r_tf.csv") as default_path:
            rec_tf_path = default_path

    if isinstance(rec_tf_path, Path):
        rec_tf_path = str(rec_tf_path)

    if tf_target_path is None:
        with importlib.resources.path("PopCILA.data", "tf_tg.csv") as default_path:
            tf_target_path = default_path

    if isinstance(tf_target_path, Path):
        tf_target_path = str(tf_target_path)

    cell_types = adata.obs[cell_type_field].unique()
    expr_prop = pd.DataFrame(index=adata.var_names)
    expr_mean = pd.DataFrame(index=adata.var_names)
    rec_tf = pd.read_csv(rec_tf_path)       # receptor, TF
    tf_target = pd.read_csv(tf_target_path)      # TF, target

    genes_in_data = set(adata.var_names)
    
    lig_rec = lig_rec[
        lig_rec['ligand'].isin(genes_in_data) &
        lig_rec['receptor'].isin(genes_in_data)
    ].copy()
    
    rec_tf = rec_tf[
        rec_tf['receptor'].isin(genes_in_data) &
        rec_tf['TF'].isin(genes_in_data)
    ].copy()
    
    tf_target = tf_target[
        tf_target['TF'].isin(genes_in_data) &
        tf_target['target'].isin(genes_in_data)
    ].copy()

    for ct in cell_types:
        adata_ct = adata[adata.obs[cell_type_field] == ct]
        expr = adata_ct.X.toarray() if hasattr(adata_ct.X, "toarray") else adata_ct.X
        expr_prop[ct] = (expr > 0).sum(axis=0) / expr.shape[0]
        expr_mean[ct] = expr.mean(axis=0)

    expr_diff = pd.DataFrame(index=expr_mean.index, columns=cell_types)
    for ct in cell_types:
        other = [c for c in cell_types if c != ct]
        expr_diff[ct] = expr_mean[ct] - expr_mean[other].mean(axis=1)
    """
    def high_expr_genes(ct):
        other = [c for c in cell_types if c != ct]
        base_expr = expr_mean[other].mean(axis=1).replace(0, 1e-6)
        cond = (
            (expr_prop[ct] > expr_prop_thresh)
            & (expr_diff[ct] > base_expr * mean_expr_diff_thresh)
        )
        return expr_prop.index[cond].tolist()
    """
    def high_expr_genes(ct):
        other = [c for c in cell_types if c != ct]
        cond = (
            (expr_prop[ct] > expr_prop_thresh)
            & (expr_diff[ct] > mean_expr_diff_thresh) 
        )
        return expr_prop.index[cond].tolist()

    high_expr_by_type = {ct: high_expr_genes(ct) for ct in cell_types}

    def fisher_activation(tf_targets, high_genes, all_genes, TF_p_adj=False):
        results = []
        for tf, tg_list in tf_targets.items():
            tg_list = [t for t in tg_list if t in all_genes]
            if len(tg_list) == 0: 
                continue
            a = len(set(tg_list) & set(high_genes))
            b = len(set(tg_list)) - a
            c = len(set(high_genes)) - a
            d = len(all_genes) - (a + b + c)
            if min(a,b,c,d) < 0: continue
            _, p = fisher_exact([[a,b],[c,d]], alternative='greater')
            results.append((tf, p))
        df = pd.DataFrame(results, columns=['TF','pval'])
        if TF_p_adj:
            from statsmodels.stats.multitest import multipletests
            reject, pvals_corrected, _, _ = multipletests(df['pval'], alpha=pval_thresh, method='fdr_bh')
            df['padj'] = pvals_corrected
            return df[df['padj'] < pval_thresh]['TF'].tolist()
        else:
            return df[df['pval'] < pval_thresh]['TF'].tolist()
    
    tf_to_target = tf_target.groupby('TF')['target'].apply(list).to_dict()
    activated_TF_by_type = {}
    for ct in cell_types:
        high_genes = high_expr_by_type[ct]
        activated_TF_by_type[ct] = fisher_activation(tf_to_target, high_genes, list(set(adata.var_names)),TF_p_adj=TF_p_adj)
        #activated_TF_by_type[ct] = fisher_activation(tf_to_target, list(set(high_genes)&set(tf_target['target'].unique())), list(set(adata.var_names)&set(tf_target['target'].unique())))

    rec_to_TF = rec_tf.groupby('receptor')['TF'].apply(list).to_dict()

    def fisher_receptor_activation(rec_to_TF, activated_TFs, all_TFs, rec_p_adj=False):
        results = []
        for r, tfs in rec_to_TF.items():
            tfs = [t for t in tfs if t in all_TFs]
            if len(tfs) == 0:
                continue   
            a = len(set(tfs) & set(activated_TFs))
            b = len(set(tfs)) - a
            c = len(set(activated_TFs)) - a
            d = len(all_TFs) - (a + b + c)
            if min(a, b, c, d) < 0:
                continue    
            _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
            results.append((r, p))
        df = pd.DataFrame(results, columns=['receptor', 'pval'])
        if rec_p_adj:
            from statsmodels.stats.multitest import multipletests
            reject, pvals_corrected, _, _ = multipletests(df['pval'], alpha=pval_thresh, method='fdr_bh')
            df['padj'] = pvals_corrected
            return df[df['padj'] < pval_thresh]['receptor'].tolist()
        else:
            return df[df['pval'] < pval_thresh]['receptor'].tolist()
        
    
    #all_TFs = list(tf_target['TF'].unique())
    #all_TFs = list(set([t for t in tf_target['TF'].unique() if pd.notna(t)] + list(rec_tf['TF'].unique())))
    all_TFs = list(set(tf_target['TF'].dropna().unique()) | set(rec_tf['TF'].dropna().unique()))
    #list(set(all_TFs) & set(adata.var_names))
    activated_receptor_by_type = {}
    for ct in cell_types:
        activated_receptor_by_type[ct] = fisher_receptor_activation(rec_to_TF, activated_TF_by_type[ct], list(set(all_TFs) & set(adata.var_names)), rec_p_adj=rec_p_adj)

    def kendall_filter(pairs, adata_ct, tau_threshold=tau_thresh, threshold=0.05):
        expr = adata_ct.to_df()
        valid_pairs = []
        for a, b in pairs:
            if a in expr.columns and b in expr.columns:
                tau, p = kendalltau(expr[a], expr[b])
                if tau>tau_threshold and p < threshold:
                    valid_pairs.append((a, b))
        return valid_pairs

    if mode == 'pre':
        filtered_pathways = []
    
        for ct in cell_types:
            adata_ct = adata[adata.obs[cell_type_field] == ct]
            active_R = activated_receptor_by_type[ct]
            active_TF = activated_TF_by_type[ct]
            if not active_R or not active_TF:
                continue
            
            # receptor–TF
            r_tf_pairs = rec_tf[rec_tf['receptor'].isin(active_R) & rec_tf['TF'].isin(active_TF)]
            valid_r_tf = kendall_filter(r_tf_pairs[['receptor','TF']].values, adata_ct)
            valid_r = [r for r, _ in valid_r_tf]
            valid_tf = [t for _, t in valid_r_tf]
            
            # ligand–receptor
            lig_r_pairs = lig_rec[lig_rec['receptor'].isin(valid_r)]
            valid_ligands = lig_r_pairs['ligand'].unique().tolist()
            
            # L–R–TF
            pathways = pd.DataFrame(
                [(l, r, t) for (l, r) in lig_r_pairs[["ligand", "receptor"]].values for (r2, t) in valid_r_tf if r == r2],
                columns=["ligand", "receptor", "TF"],
            )
            pathways["receiver"] = ct
            filtered_pathways.append(pathways)
        if len(filtered_pathways) == 0:
            final_pathways = pd.DataFrame(columns=['ligand','receptor','TF','receiver'])
        else:
            final_pathways = pd.concat(filtered_pathways, ignore_index=True)
                
    elif mode == 'after':
        filtered_pathways_list = []
        
        for ct in cell_types:  
            adata_ct = adata[adata.obs[cell_type_field] == ct]
            active_R = activated_receptor_by_type[ct]
            active_TF = activated_TF_by_type[ct]
            if not active_R or not active_TF:
                continue
        
            candidate_df = lig_rec[
                lig_rec['receptor'].isin(active_R) &
                lig_rec['TF'].isin(active_TF)
            ].copy()
        
            if candidate_df.empty:
                continue

            pairs = list(candidate_df[['receptor', 'TF']].itertuples(index=False, name=None))
            valid_r_tf = kendall_filter(pairs, adata_ct)
            if len(valid_r_tf) == 0:
                continue

            valid_r_tf_set = set(valid_r_tf)
            filtered = candidate_df[
                candidate_df.apply(lambda x: (x['receptor'], x['TF']) in valid_r_tf_set, axis=1)
            ].copy()

            if not filtered.empty:
                filtered['receiver'] = ct
                filtered_pathways_list.append(filtered)

        if len(filtered_pathways_list) == 0:
            final_pathways = pd.DataFrame(columns=['ligand', 'receptor', 'TF', 'receiver'])
        else:
            final_pathways = pd.concat(filtered_pathways_list, ignore_index=True)
            final_pathways = add_sender_from_CCIdetect_wide(final_pathways, cci_pvalues, p_threshold=pval_thresh)[['sender','ligand', 'receiver','receptor', 'TF']]

    if final_pathways.empty:
        L_list, R_list, TF_list = [], [], []
    else:
        L_list = final_pathways['ligand'].unique().tolist()
        R_list = final_pathways['receptor'].unique().tolist()
        TF_list = final_pathways['TF'].unique().tolist()


    active_lists = {
        'ligand': L_list,
        'receptor': R_list,
        'TF': TF_list,
    }

    return final_pathways, active_lists

def find_spatial_LRTF_links(
    adata,
    res: pd.DataFrame,
    rectf_path: str = None,
    moranI_threshold: float = 0.0,
    k: int = 8,
    permutations: int = 999
) -> pd.DataFrame:

    if rectf_path is None:
        with importlib.resources.path("PopCILA.data", "r_tf.csv") as default_path:
            rectf_path = default_path

    if isinstance(rectf_path, Path):
        rectf_path = str(rectf_path)

    rectf_df = pd.read_csv(rectf_path)

    print(f"Building spatial neighbors (k={k}) ...")
    sq.gr.spatial_neighbors(adata, n_neighs=k, coord_type='generic')

    lr_genes = set(res['ligand']) | set(res['receptor'])
    rt_genes = set(rectf_df['receptor']) | set(rectf_df['TF'])
    candidate_genes = lr_genes | rt_genes
    candidate_genes = [g for g in candidate_genes if g in adata.var_names]

    if len(candidate_genes) == 0:
        print("No candidate genes found in adata.var_names, returning empty DataFrame.")
        return pd.DataFrame(columns=[
            'ligand', 'receptor', 'TF',
            'I_biv_LR', 'p_LR', 'p_adj_LR',
            'I_biv_RT', 'p_RT', 'p_adj_RT'
        ])

    print(f"Calculating global Moran's I for {len(candidate_genes)} candidate genes ...")
    sq.gr.spatial_autocorr(
        adata,
        mode="moran",
        genes=candidate_genes,
        n_perms=permutations,
        #n_jobs=4,
    )

    moran_key = 'moranI' if 'moranI' in adata.uns else 'moran'
    moran_df = pd.DataFrame(adata.uns[moran_key])

    if 'pval_sim_fdr_bh' in moran_df.columns:
        p_adj = moran_df['pval_sim_fdr_bh']
        p_raw = moran_df['pval_sim']
    else:
        p_adj = moran_df['pval_norm_fdr_bh']
        p_raw = moran_df['pval_norm']

    moran_df['p_adj'] = p_adj
    moran_df['p_raw'] = p_raw

    spatial_genes = moran_df.query(
        f'I > {moranI_threshold} and p_adj < 0.05'
    ).index.tolist()
    print(f"Retained {len(spatial_genes)} spatially variable genes (I > {moranI_threshold}).")

    ligrec_df = res.copy()
    ligrec_df = ligrec_df[
        ligrec_df['ligand'].isin(spatial_genes) &
        ligrec_df['receptor'].isin(spatial_genes)
    ].reset_index(drop=True)

    valid_receptors = ligrec_df['receptor'].unique()
    rectf_df = rectf_df[
        rectf_df['receptor'].isin(valid_receptors) &
        rectf_df['TF'].isin(spatial_genes)
    ].reset_index(drop=True)

    print(f"Filtered LR pairs: {len(ligrec_df)}, RT pairs: {len(rectf_df)}")

    if len(ligrec_df) == 0 or len(rectf_df) == 0:
        print("No LR or RT pairs left after spatial filtering, returning empty DataFrame.")
        return pd.DataFrame(columns=[
            'ligand', 'receptor', 'TF',
            'I_biv_LR', 'p_LR', 'p_adj_LR',
            'I_biv_RT', 'p_RT', 'p_adj_RT'
        ])

    coords = adata.obsm['spatial']
    w = libpysal.weights.KNN.from_array(coords, k=k)
    w.transform = 'r'

    genes_for_bv = sorted(
        set(ligrec_df['ligand']) |
        set(ligrec_df['receptor']) |
        set(rectf_df['TF'])
    )
    genes_for_bv = [g for g in genes_for_bv if g in adata.var_names]

    expr = adata[:, genes_for_bv].X
    if issparse(expr):
        expr = expr.toarray()
    expr = np.asarray(expr, dtype=float)    # (n_spots, n_genes)

    gene_to_idx = {g: i for i, g in enumerate(genes_for_bv)}

    def calc_bivariate_moran_fast(gene_a, gene_b, expr, gene_to_idx, w, permutations):
        if gene_a not in gene_to_idx or gene_b not in gene_to_idx:
            return np.nan, np.nan

        x = expr[:, gene_to_idx[gene_a]]
        y = expr[:, gene_to_idx[gene_b]]

        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan, np.nan

        try:
            mbv = Moran_BV(x, y, w, permutations=permutations)
            return mbv.I, mbv.p_sim
        except Exception:
            return np.nan, np.nan

    print("Computing Bivariate Moran's I for ligand–receptor pairs ...")
    LR_results = []
    for _, row in ligrec_df.iterrows():
        I, p = calc_bivariate_moran_fast(
            row['ligand'], row['receptor'],
            expr, gene_to_idx, w, permutations
        )
        LR_results.append([row['ligand'], row['receptor'], I, p])

    LR_df = pd.DataFrame(LR_results, columns=['ligand', 'receptor', 'I_biv_LR', 'p_LR'])
    LR_df = LR_df.dropna()
    if LR_df.shape[0] == 0:
        print("No valid LR Moran_BV results, returning empty DataFrame.")
        return pd.DataFrame(columns=[
            'ligand', 'receptor', 'TF',
            'I_biv_LR', 'p_LR', 'p_adj_LR',
            'I_biv_RT', 'p_RT', 'p_adj_RT'
        ])

    LR_df['p_adj_LR'] = multipletests(LR_df['p_LR'], method='fdr_bh')[1]
    LR_df = LR_df.query(f'I_biv_LR > {moranI_threshold} and p_adj_LR < 0.05')
    print(f"{len(LR_df)} significant LR pairs retained.")

    if LR_df.shape[0] == 0:
        print("No significant LR pairs after FDR, returning empty DataFrame.")
        return pd.DataFrame(columns=[
            'ligand', 'receptor', 'TF',
            'I_biv_LR', 'p_LR', 'p_adj_LR',
            'I_biv_RT', 'p_RT', 'p_adj_RT'
        ])
    
    sig_receptors = LR_df['receptor'].unique()
    rectf_df = rectf_df[rectf_df['receptor'].isin(sig_receptors)].reset_index(drop=True)
    print(f"Restrict RT pairs to receptors from significant LR pairs: {len(rectf_df)} RT pairs left.")

    print("Computing Bivariate Moran's I for receptor–TF pairs ...")
    RT_results = []
    for _, row in rectf_df.iterrows():
        I, p = calc_bivariate_moran_fast(
            row['receptor'], row['TF'],
            expr, gene_to_idx, w, permutations
        )
        RT_results.append([row['receptor'], row['TF'], I, p])

    RT_df = pd.DataFrame(RT_results, columns=['receptor', 'TF', 'I_biv_RT', 'p_RT'])
    RT_df = RT_df.dropna()
    if RT_df.shape[0] == 0:
        print("No valid RT Moran_BV results, returning empty DataFrame.")
        return pd.DataFrame(columns=[
            'ligand', 'receptor', 'TF',
            'I_biv_LR', 'p_LR', 'p_adj_LR',
            'I_biv_RT', 'p_RT', 'p_adj_RT'
        ])

    RT_df['p_adj_RT'] = multipletests(RT_df['p_RT'], method='fdr_bh')[1]
    RT_df = RT_df.query(f'I_biv_RT > {moranI_threshold} and p_adj_RT < 0.05')
    print(f"{len(RT_df)} significant RT pairs retained.")

    if RT_df.shape[0] == 0:
        print("No significant RT pairs after FDR, returning empty DataFrame.")
        return pd.DataFrame(columns=[
            'ligand', 'receptor', 'TF',
            'I_biv_LR', 'p_LR', 'p_adj_LR',
            'I_biv_RT', 'p_RT', 'p_adj_RT'
        ])

    LR_TF_links = LR_df.merge(RT_df, on='receptor', how='inner')
    print(f"Total {len(LR_TF_links)} ligand–receptor–TF chains identified.")

    return LR_TF_links[
        ['ligand', 'receptor', 'TF',
         'I_biv_LR', 'p_LR', 'p_adj_LR',
         'I_biv_RT', 'p_RT', 'p_adj_RT']
    ]


def plot_LRTF_network(df,
                      title="Sender-Ligand-Receiver-Receptor-TF Network",
                      figsize=(12, 10),
                      node_colors=None,
                      fontsize=9,
                      edge_color='#808080',
                      return_graph=False,
                      node_size=800,
                      show_role_in_label=False,
                      savefig=None,
                      dpi=300):

    if node_colors is None:
        node_colors = {
            'sender': '#FF6B6B',
            'ligand': "#4ECDC4",
            'receiver': "#45B7D1",
            'receptor': "#96CEB4",
            'TF': "#FFEAA7"
        }

    G = nx.DiGraph()

    def node_id(role, name):
        return f"{role}::{name}"

    labels = {}

    for _, row in df.iterrows():
        pair = f"{row['sender']}->{row['receiver']}"
        s   = node_id("sender",   row["sender"])
        l   = node_id("ligand",   f"{row['ligand']}||{pair}")
        rcv = node_id("receiver", f"{row['receiver']}||{pair}")
        rcp = node_id("receptor", f"{row['receptor']}||{pair}")
        tf  = node_id("TF",       f"{row['TF']}||{pair}")

        G.add_node(s, type="sender")
        G.add_node(l, type="ligand")
        G.add_node(rcv, type="receiver")
        G.add_node(rcp, type="receptor")
        G.add_node(tf, type="TF")

        labels[s]   = f"{row['sender']}\n(sender)"     if show_role_in_label else str(row["sender"])

        labels[l]   = f"{row['ligand']}\n(ligand)"     if show_role_in_label else str(row["ligand"])
        labels[rcv] = f"{row['receiver']}\n(receiver)" if show_role_in_label else str(row["receiver"])
        labels[rcp] = f"{row['receptor']}\n(receptor)" if show_role_in_label else str(row["receptor"])
        labels[tf]  = f"{row['TF']}\n(TF)"             if show_role_in_label else str(row["TF"])

        G.add_edge(s, l)
        G.add_edge(l, rcv)
        G.add_edge(rcv, rcp)
        G.add_edge(rcp, tf)

    sender_nodes   = [n for n, d in G.nodes(data=True) if d['type'] == 'sender']
    ligand_nodes   = [n for n, d in G.nodes(data=True) if d['type'] == 'ligand']
    receiver_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'receiver']
    receptor_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'receptor']
    tf_nodes       = [n for n, d in G.nodes(data=True) if d['type'] == 'TF']

    shells = [sender_nodes, ligand_nodes, receiver_nodes, receptor_nodes, tf_nodes]
    pos = nx.shell_layout(G, nlist=shells)

    node_edge_col = "#B0B0B0"
    node_edge_lw  = 0.8

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_color,
        alpha=0.7,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=18,
        width=1.4,
        min_source_margin=10,
        min_target_margin=14,
        connectionstyle="arc3,rad=0.06"
    )

    nx.draw_networkx_nodes(G, pos, nodelist=sender_nodes,
                           node_color=node_colors['sender'], node_shape='s',
                           node_size=node_size, edgecolors=node_edge_col, linewidths=node_edge_lw, ax=ax)

    nx.draw_networkx_nodes(G, pos, nodelist=ligand_nodes,
                           node_color=node_colors['ligand'], node_shape='o',
                           node_size=node_size, edgecolors=node_edge_col, linewidths=node_edge_lw, ax=ax)

    nx.draw_networkx_nodes(G, pos, nodelist=receiver_nodes,
                           node_color=node_colors['receiver'], node_shape='s',
                           node_size=node_size, edgecolors=node_edge_col, linewidths=node_edge_lw, ax=ax)

    nx.draw_networkx_nodes(G, pos, nodelist=receptor_nodes,
                           node_color=node_colors['receptor'], node_shape='o',
                           node_size=node_size, edgecolors=node_edge_col, linewidths=node_edge_lw, ax=ax)

    nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes,
                           node_color=node_colors['TF'], node_shape='o',
                           node_size=node_size, edgecolors=node_edge_col, linewidths=node_edge_lw, ax=ax)

    texts = [
        ax.text(pos[n][0], pos[n][1], labels.get(n, n),
                fontsize=fontsize, ha='center', va='center')
        for n in G.nodes()
    ]
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.3), ax=ax)

    ax.set_title(title, fontsize=14, weight='bold')
    ax.axis('off')

    legend_elements = [Patch(facecolor=c, edgecolor='black', label=t) for t, c in node_colors.items()]
    ax.legend(handles=legend_elements, loc='upper center', ncol=len(node_colors))

    plt.tight_layout()

    if savefig is not None:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()

    if return_graph:
        return G
    else:
        return ax

def plot_LRTF_network_sankey(df, title=" ",fontsize=14, width=1200, height=1200):
    all_nodes = []
    columns = ['sender', 'ligand', 'receiver', 'receptor', 'TF']
    node_indices = {}
    node_labels = []
    node_colors = []

    role_colors = {
        'sender': '#FF6B6B', 'ligand': "#4ECDC4", 
        'receiver': "#45B7D1", 'receptor': "#96CEB4", 'TF': "#FFEAA7"
    }

    counter = 0
    for col in columns:
        unique_vals = df[col].unique()
        for val in unique_vals:
            key = f"{col}_{val}"
            if key not in node_indices:
                node_indices[key] = counter
                node_labels.append(f"{val}")
                node_colors.append(role_colors[col])
                counter += 1

    sources = []
    targets = []
    values = []
    
    path_steps = [('sender', 'ligand'), 
                  ('ligand', 'receiver'), 
                  ('receiver', 'receptor'), 
                  ('receptor', 'TF')]

    for i, row in df.iterrows():
        weight = 1 
        
        for src_col, tgt_col in path_steps:
            src_key = f"{src_col}_{row[src_col]}"
            tgt_key = f"{tgt_col}_{row[tgt_col]}"
            
            sources.append(node_indices[src_key])
            targets.append(node_indices[tgt_key])
            values.append(weight)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            # color='rgba(200, 200, 200, 0.3)' 
        )
    )])

    fig.update_layout(title_text=title, font_size=fontsize, width=width, height=height)
    fig.show()

def plot_LRTF_network_sunburst(
    df,
    cols=("sender", "ligand", "receiver", "receptor", "TF"),
    weight_col=None,
    layer_colors=None,
    title=" ",
    width=900,
    height=900,
    margin=(60, 10, 10, 10),
    font_family="Arial",
    fontsize=14,
    title_fontsize=18,
    hover_fontsize=14,
    uniformtext_minsize=9,
    uniformtext_mode="hide",
    savepath=None,
    scale=2
):
    d = df.loc[:, cols].copy()
    for c in cols:
        d[c] = d[c].astype(str)

    if weight_col is None:
        d["_w"] = 1
        weight_col = "_w"

    g = d.groupby(list(cols), as_index=False)[weight_col].sum()

    if layer_colors is None:
        """
        layer_colors = [
            "rgba(255, 179, 186, 0.95)",
            "rgba(186, 225, 255, 0.95)",
            "rgba(186, 255, 201, 0.95)",
            "rgba(255, 255, 186, 0.95)",
            "rgba(221, 204, 255, 0.95)",
        ]
        """
        layer_colors = [
            "rgba(204, 102, 002, 0.5)",
            "rgba(009, 147, 150, 0.5)",
            "rgba(255, 179, 186, 1)",
            "rgba(255, 183, 003, 0.5)",
            "rgba(221, 204, 255, 1)",
        ]
    assert len(layer_colors) == len(cols)

    node_value, node_depth, node_label, node_parent = {}, {}, {}, {}

    for _, row in g.iterrows():
        path = [row[c] for c in cols]
        w = float(row[weight_col])
        for depth in range(len(cols)):
            nid = "||".join(path[:depth+1])
            parent = "" if depth == 0 else "||".join(path[:depth])
            node_value[nid] = node_value.get(nid, 0.0) + w
            node_depth[nid] = depth
            node_label[nid] = path[depth]
            node_parent[nid] = parent

    nodes_sorted = sorted(node_value.keys(), key=lambda k: (node_depth[k], k))

    ids, parents, labels, values, colors, customdata = [], [], [], [], [], []
    for nid in nodes_sorted:
        depth = node_depth[nid]
        ids.append(nid)
        parents.append(node_parent[nid])
        labels.append(node_label[nid])
        values.append(node_value[nid])
        colors.append(layer_colors[depth])
        customdata.append(cols[depth])

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=colors, line=dict(width=1.2, color="white")),
        insidetextorientation="radial",
        customdata=customdata,
        hovertemplate="<b>%{label}</b><br>level=%{customdata}<br>value=%{value}<extra></extra>"
    ))

    t, l, r, b = margin
    fig.update_layout(
        title=dict(text=title, font=dict(size=title_fontsize, family=font_family)),
        width=width,
        height=height,
        margin=dict(t=t, l=l, r=r, b=b),
        font=dict(size=fontsize, family=font_family),
        hoverlabel=dict(font_size=hover_fontsize, font_family=font_family),
        uniformtext=dict(minsize=uniformtext_minsize, mode=uniformtext_mode)
    )

    if savepath is not None:
        fig.write_image(savepath, width=width, height=height, scale=scale)

    fig.show()
    #return fig


def plot_SpatialPathway(
    adata,
    lr_tf_triplet: tuple[str, str, str],
    layer: str = None,
    topn_frac: float = 0.2,
    knn: int = 8,
    pt_size: float = 2.0,
    alpha_min: float = 0.1,
    max_cut: float = 0.95,
    clip_pathway: bool = True,
    subset_key: str = None,
    subset_condition: str = None,
    figsize: tuple = (14, 6),
    dual_plot: bool = True
):
    """
    Visualize the spatial pathway activity of a ligand–receptor–TF chain
    (computed on subset, plotted on all spots).

    Pathway activity = ((LR activity) * (max TF expression in neighborhood)) ** (1/3)
    """

    ligand, receptor, tf = lr_tf_triplet
    for g in [ligand, receptor, tf]:
        if g not in adata.var_names:
            raise ValueError(f"Gene {g} not found in adata.var_names")

    if subset_key and subset_key in adata.obs:
        query_str = f"{subset_key} {subset_condition}" if subset_condition else None
        adata_sub = adata[adata.obs.eval(query_str)].copy() if query_str else adata.copy()
        print(f"Using subset: {query_str}, {adata_sub.shape[0]} spots retained.")
    else:
        adata_sub = adata.copy()
        print(f"Using all {adata_sub.shape[0]} spots for calculation.")

    expr = adata_sub.layers[layer] if layer else adata_sub.X
    if issparse(expr):
        expr = expr.toarray()
    coords_sub = adata_sub.obsm.get("spatial")
    coords_all = adata.obsm.get("spatial")
    if coords_all is None:
        raise KeyError("Spatial coordinates not found in adata.obsm['spatial']")

    nn_model = NearestNeighbors(n_neighbors=knn + 1).fit(coords_sub)
    _, nn_indices = nn_model.kneighbors(coords_sub)

    lig = expr[:, adata_sub.var_names.get_loc(ligand)]
    rec = expr[:, adata_sub.var_names.get_loc(receptor)]
    tf_expr = expr[:, adata_sub.var_names.get_loc(tf)]

    n_spots = expr.shape[0]
    neighbor_expr = np.zeros((3, n_spots))
    for i in range(n_spots):
        neighbors = nn_indices[i, 1:]
        neighbor_expr[0, i] = np.max(lig[neighbors])
        neighbor_expr[1, i] = np.max(rec[neighbors])
        neighbor_expr[2, i] = np.max(tf_expr[neighbors])

    lr_activity = np.maximum(lig * neighbor_expr[1], rec * neighbor_expr[0])
    lr_activity = np.clip(lr_activity, None, np.quantile(lr_activity, max_cut)) if clip_pathway else lr_activity
    pathway_activity = np.cbrt(lr_activity * neighbor_expr[2])
    if clip_pathway:
        path_cut = np.quantile(pathway_activity, max_cut)
        pathway_activity = np.clip(pathway_activity, None, path_cut)
        print(f"Clipped pathway activity at {max_cut} quantile: {path_cut:.3f}")

    # Initialize all scores as NaN, then fill subset
    activity_all = pd.Series(np.nan, index=adata.obs.index)
    activity_all.loc[adata_sub.obs.index] = pathway_activity

    topn = int(topn_frac * n_spots)
    def get_top_idx(gene_expr):
        order = np.argsort(-gene_expr + np.random.randn(n_spots) * 1e-6)
        return order[:topn] if np.sum(gene_expr > 0) >= topn else np.where(gene_expr > 0)[0]

    lig_high = get_top_idx(lig)
    rec_high = get_top_idx(rec)
    tf_high  = get_top_idx(tf_expr)

    exp_type = np.zeros(n_spots, dtype=int)
    exp_type[lig_high] = 1
    exp_type[rec_high] = 2
    exp_type[tf_high] = 3
    exp_type[np.intersect1d(lig_high, rec_high)] = 4
    exp_type[np.intersect1d(rec_high, tf_high)] = 5
    exp_type[np.intersect1d(lig_high, tf_high)] = 6
    exp_type[np.intersect1d(np.intersect1d(lig_high, rec_high), tf_high)] = 7

    plot_df_sub = pd.DataFrame({
        "x": coords_sub[:, 0],
        "y": coords_sub[:, 1],
        "activity": pathway_activity,
        "type": exp_type
    }, index=adata_sub.obs.index)

    single_width = figsize[0] / 2
    if dual_plot:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(single_width * 1.2, figsize[1]))

    # Left: expression highlights (subset)
    if dual_plot:
        ax1.scatter(coords_all[:, 0], coords_all[:, 1], color="gray", alpha=0.2, s=pt_size)
        colors = ['gray', 'red', 'green', 'orange', 'blue', 'purple', 'black', 'gold']
        labels = [
            "All low", "Ligand high", "Receptor high", "TF high",
            "L+R high", "R+TF high", "L+TF high", "All high"
        ]
        for i in np.unique(exp_type):
            mask = plot_df_sub["type"] == i
            ax1.scatter(
                plot_df_sub.loc[mask, "x"],
                plot_df_sub.loc[mask, "y"],
                c=colors[i],
                label=labels[i],
                s=pt_size
            )
        ax1.legend(fontsize=6, loc="best")
        ax1.set_title(f"{ligand}–{receptor}–{tf} Expression")
        ax1.invert_yaxis()

    # Right: pathway activity (on all)
    alpha = (activity_all.fillna(0) - np.nanmin(activity_all)) / (
        np.nanmax(activity_all) - np.nanmin(activity_all)
    ) * (1 - alpha_min) + alpha_min

    ax2.scatter(coords_all[:, 0], coords_all[:, 1], color="gray", alpha=0.1, s=pt_size)
    scatter = ax2.scatter(
        adata_sub.obsm["spatial"][:, 0],
        adata_sub.obsm["spatial"][:, 1],
        c=pathway_activity,
        cmap="coolwarm",
        s=pt_size,
        alpha=alpha.loc[adata_sub.obs.index]
    )
    plt.colorbar(scatter, ax=ax2, label="Pathway Activity", shrink=0.8)
    ax2.set_title(f"{ligand}–{receptor}–{tf} Pathway Activity")
    ax2.invert_yaxis()

    for ax in [ax1, ax2] if dual_plot else [ax2]:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")
    plt.tight_layout()
    return fig

def compute_spatial_activity(
    adata,
    lr_pairs: list[tuple[str, str]] = None,
    lrtf_triplets: list[tuple[str, str, str]] = None,
    layer: str = None,
    knn: int = 8,
    max_cut: float = 0.95,
    clip_activity: bool = False,
    subset_key: str = None,
    subset_condition: str = None,
    write_to_adata: bool = True,
    fill_value: str = "NaN"
):
    """
    Batch compute spatial activity scores for multiple LR pairs or LRTF triplets,
    and optionally write results to adata.obs.

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data.
    lr_pairs : list of tuple[str, str], optional
        List of ligand–receptor pairs.
    lrtf_triplets : list of tuple[str, str, str], optional
        List of ligand–receptor–TF triplets.
    layer : str, optional
        Matrix layer to use (default: adata.X).
    knn : int, optional
        Number of nearest neighbors (default: 8).
    max_cut : float, optional
        Quantile cutoff for clipping extreme values (default: 0.95).
    clip_activity : bool, optional
        Whether to clip extreme values.
    subset_key : str, optional
        Column in adata.obs for subsetting (e.g. 'refined_beta').
    subset_condition : str, optional
        Condition string for subsetting (e.g. '>0').
    write_to_adata : bool, optional
        If True, write each activity column into adata.obs.
    fill_value : str, optional
        How to fill scores for unselected spots: "NaN" or "zero".

    Returns
    -------
    pd.DataFrame
        DataFrame with spatial coordinates (x, y) and activity scores.
    """

    if (lr_pairs is None) == (lrtf_triplets is None):
        raise ValueError("Provide either lr_pairs or lrtf_triplets, not both.")

    if subset_key and subset_key in adata.obs:
        query_str = f"{subset_key} {subset_condition}" if subset_condition else None
        adata_sub = adata[adata.obs.eval(query_str)].copy() if query_str else adata.copy()
        print(f"Using subset: {query_str}, {adata_sub.shape[0]} spots retained.")
    else:
        adata_sub = adata.copy()
        print(f"Using all {adata_sub.shape[0]} spots for calculation.")

    expr = adata_sub.layers[layer] if layer else adata_sub.X
    if issparse(expr):
        expr = expr.toarray()

    coords = adata_sub.obsm.get("spatial")
    if coords is None:
        raise KeyError("Spatial coordinates not found in adata.obsm['spatial']")

    nn_model = NearestNeighbors(n_neighbors=knn + 1).fit(coords)
    _, nn_indices = nn_model.kneighbors(coords)
    n_spots = expr.shape[0]

    results = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1]},
        index=adata_sub.obs.index
    )

    if lr_pairs is not None:
        for ligand, receptor in lr_pairs:
            if ligand not in adata_sub.var_names or receptor not in adata_sub.var_names:
                print(f"Skip {ligand}-{receptor}: gene not found.")
                continue

            lig = expr[:, adata_sub.var_names.get_loc(ligand)]
            rec = expr[:, adata_sub.var_names.get_loc(receptor)]

            if (np.nanmax(lig) <= 0) and (np.nanmax(rec) <= 0):
                print(f"Skip {ligand}-{receptor}: ligand & receptor expression all ≤ 0 in subset.")
                continue

            neighbor_expr = np.zeros((2, n_spots))
            for i in range(n_spots):
                neighbors = nn_indices[i, 1:]
                neighbor_expr[0, i] = np.max(lig[neighbors])
                neighbor_expr[1, i] = np.max(rec[neighbors])

            lr_activity = np.maximum(lig * neighbor_expr[1], rec * neighbor_expr[0])

            if np.all(np.isnan(lr_activity)) or np.nanmax(lr_activity) <= 0:
                print(f"Skip {ligand}-{receptor}: raw LR activity all zero/NaN.")
                continue

            if clip_activity:
                positive_vals = lr_activity[lr_activity > 0]
                if positive_vals.size > 0:
                    lr_cut = np.quantile(positive_vals, max_cut)
                    lr_activity = np.clip(lr_activity, None, lr_cut)
                    print(
                        f"Computed LR activity for {ligand}-{receptor} "
                        f"(clipped at {max_cut:.2f} quantile={lr_cut:.4g})."
                    )
                else:
                    print(
                        f"No positive LR activity values for {ligand}-{receptor}; "
                        f"skipping clipping, using raw activity."
                    )
            else:
                print(f"Computed LR activity for {ligand}-{receptor} (raw).")

            col_name = f"{ligand}_{receptor}_score"
            results[col_name] = lr_activity

    else:
        for ligand, receptor, tf in lrtf_triplets:
            missing = [g for g in [ligand, receptor, tf] if g not in adata_sub.var_names]
            if missing:
                print(f"Skip {ligand}-{receptor}-{tf}: missing genes {missing}.")
                continue

            lig = expr[:, adata_sub.var_names.get_loc(ligand)]
            rec = expr[:, adata_sub.var_names.get_loc(receptor)]
            tf_expr = expr[:, adata_sub.var_names.get_loc(tf)]

            if (np.nanmax(lig) <= 0) and (np.nanmax(rec) <= 0) and (np.nanmax(tf_expr) <= 0):
                print(f"Skip {ligand}-{receptor}-{tf}: ligand/receptor/TF expression all ≤ 0 in subset.")
                continue

            neighbor_expr = np.zeros((3, n_spots))
            for i in range(n_spots):
                neighbors = nn_indices[i, 1:]
                neighbor_expr[0, i] = np.max(lig[neighbors])
                neighbor_expr[1, i] = np.max(rec[neighbors])
                neighbor_expr[2, i] = np.max(tf_expr[neighbors])

            lr_activity = np.maximum(lig * neighbor_expr[1], rec * neighbor_expr[0])

            if np.all(np.isnan(lr_activity)) or np.nanmax(lr_activity) <= 0:
                print(f"Skip {ligand}-{receptor}-{tf}: raw LR activity all zero/NaN.")
                continue

            if clip_activity:
                lr_pos = lr_activity[lr_activity > 0]
                if lr_pos.size > 0:
                    lr_cut = np.quantile(lr_pos, max_cut)
                    lr_activity = np.clip(lr_activity, None, lr_cut)
                else:
                    print(
                        f"No positive LR activity values for {ligand}-{receptor}-{tf}; "
                        f"skipping LR clipping."
                    )

            tf_neighbor_max = neighbor_expr[2]
            pathway_activity = np.sqrt(lr_activity * tf_neighbor_max)
            # pathway_activity = np.cbrt(lr_activity * tf_neighbor_max)
            # pathway_activity = lr_activity * tf_neighbor_max

            if np.all(np.isnan(pathway_activity)) or np.nanmax(pathway_activity) <= 0:
                print(f"Skip {ligand}-{receptor}-{tf}: raw pathway activity all zero/NaN.")
                continue

            if clip_activity:
                path_pos = pathway_activity[pathway_activity > 0]
                if path_pos.size > 0:
                    path_cut = np.quantile(path_pos, max_cut)
                    pathway_activity = np.clip(pathway_activity, None, path_cut)
                    print(
                        f"Computed LRTF activity for {ligand}-{receptor}-{tf} "
                        f"(clipped at {max_cut:.2f} quantile={path_cut:.4g})."
                    )
                else:
                    print(
                        f"No positive pathway activity values for {ligand}-{receptor}-{tf}; "
                        f"skipping pathway clipping, using raw values."
                    )
            else:
                print(f"Computed LRTF activity for {ligand}-{receptor}-{tf} (raw).")

            col_name = f"{ligand}_{receptor}_{tf}_score"
            results[col_name] = pathway_activity

    if write_to_adata:
        print("\nWriting scores to adata.obs ...")
        for col in results.columns[2:]:
            adata.obs[col] = np.nan
            adata.obs.loc[results.index, col] = results[col].values
            if fill_value.lower() == "nan":
                adata.obs[col] = adata.obs[col].replace(0, np.nan)

    print(f"\nDone. Calculated {results.shape[1]-2} activity columns for {results.shape[0]} spots.")
    return results
