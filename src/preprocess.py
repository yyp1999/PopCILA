#preprocess.py
"""
    Some preprocess function
"""
import pandas as pd
import numpy as np
import importlib.resources
from pathlib import Path

def filter_Bulkdata(bulk_data, lr_db_path=None, threshold=0, log=False):
    """
    Process Bulk data and return filtered L-R pairs, expression matrix, and filtered log-transformed data.

    Parameters:
    bulk_data (DataFrame): Bulk data, with samples as rows and genes as columns.
    lr_db_path (str): Path to Ligand-Receptor database. If None, use the default database in the package.
    threshold (float): Filtering threshold for gene expression, default is 0.
    log(boolean): Indicating whether to convert the gene expression data logarithm (np.log2(x + 1)), which defaults to False

    Returns:
    lr_expr_pairs (pd.DataFrame): Filtered L-R pairs.
    lr_expr_df (pd.DataFrame): Filtered expression matrix.
    """
    # If lr_db_path is not specified, use the default database in the package
    if lr_db_path is None:
        with importlib.resources.path("Chunk.data", "Human-2020-Cabello-Aguilar-LR-pairs.csv") as default_path:
            lr_db_path = default_path

    # Filter by average gene expression
    filtered_data_log = bulk_data[bulk_data.mean(axis=1) > threshold]
    #filtered_data_log = filtered_data.applymap(lambda x: np.log2(x + 1))
    if log:
        filtered_data_log = filtered_data_log.map(lambda x: np.log2(x + 1))

    # Read Ligand-Receptor database
    lr = pd.read_csv(lr_db_path)

    # Filter data to generate detectable L-R pairs in Bulk data
    expressed_pairs = []
    expressed_data = []
    for _, row in lr.iterrows():
        ligand = row['ligand']
        receptor = row['receptor']
        
        # Check if ligand and receptor both are in the expression matrix
        if ligand in filtered_data_log.index and receptor in filtered_data_log.index:
            expressed_data.append(filtered_data_log.loc[ligand])
            expressed_data.append(filtered_data_log.loc[receptor])
            expressed_pairs.append(row)

    lr_expr_df = pd.DataFrame(expressed_data)
    lr_expr_pairs = pd.DataFrame(expressed_pairs)
    lr_expr_df = lr_expr_df[~lr_expr_df.index.duplicated(keep='first')]
    lr_expr_pairs['l-r'] = lr_expr_pairs['ligand'] + '-' + lr_expr_pairs['receptor']
    lr_expr_pairs = lr_expr_pairs.drop_duplicates(subset=["l-r"], keep="first")

    return lr_expr_pairs, lr_expr_df

def filter_Bulkdata_LRTF(bulk_data, lr_df=None, threshold=0, log=False):
    """
    Process Bulk data and return filtered L-R-TF triplets, expression matrix, and filtered log-transformed data.

    Parameters:
    bulk_data (DataFrame): Bulk data, with genes as rows and samples as columns.
    lr_df (DataFrame): DataFrame containing ligand, receptor, TF columns.
    threshold (float): Filtering threshold for gene expression, default is 0.
    log (bool): Whether to log-transform the gene expression data (np.log2(x + 1)), default False.

    Returns:
    lrtf_expr_triplets (DataFrame): Filtered L-R-TF triplets.
    lrtf_expr_df (DataFrame): Filtered expression matrix containing L, R, TF genes.
    """
    if lr_df is None:
        raise ValueError("Please provide a DataFrame containing ligand, receptor, and TF columns.")

    # Filter bulk data by average gene expression
    filtered_data = bulk_data.loc[bulk_data.mean(axis=1) > threshold]
    if log:
        filtered_data = np.log2(filtered_data + 1)

    # Filter data to generate detectable L-R-TF triplets in bulk data
    expressed_triplets = []
    expressed_data = []

    for _, row in lr_df.iterrows():
        ligand = row['ligand']
        receptor = row['receptor']
        tf = row['TF']

        # Check if ligand, receptor, and TF all exist in bulk data
        if all(gene in filtered_data.index for gene in [ligand, receptor, tf]):
            expressed_data.extend([filtered_data.loc[ligand],
                                   filtered_data.loc[receptor],
                                   filtered_data.loc[tf]])
            expressed_triplets.append(row)

    lrtf_expr_df = pd.DataFrame(expressed_data)
    lrtf_expr_df = lrtf_expr_df[~lrtf_expr_df.index.duplicated(keep='first')]

    lrtf_expr_triplets = pd.DataFrame(expressed_triplets)
    lrtf_expr_triplets['l-r-tf'] = (lrtf_expr_triplets['ligand'] + '-' +
                                    lrtf_expr_triplets['receptor'] + '-' +
                                    lrtf_expr_triplets['TF'])
    lrtf_expr_triplets = lrtf_expr_triplets.drop_duplicates(subset=["l-r-tf"], keep="first")[['ligand','receptor','TF','l-r-tf']]

    return lrtf_expr_triplets, lrtf_expr_df

def filter_scRNAdata(lr, adata, label):
    """
    Process single-cell data to extract corresponding LR pairs and genes based on effective LRIs from bulk data.

    Parameters:
    lr (Dataframe): list of phenotype-specific LRIs.
    adata (Anndata): Preprocessed single-cell data provided by the user, recommended in TPM format.
    label (str): Column name in adata.obs for cell type or ID.

    Returns:
    lr_expr_df (pd.DataFrame): Expression matrix of corresponding LR genes in single-cell data.
    lr_expr_pairs (pd.DataFrame): List of corresponding LR pairs in single-cell data.
    """
    unique_genes = sorted(set(lr['ligand'].unique()) | set(lr['receptor'].unique()))
    
    genes_in_adata = [gene for gene in unique_genes if gene in adata.var_names]
    if len(genes_in_adata) < len(unique_genes):
        print(f"Warning: {len(unique_genes) - len(genes_in_adata)} genes not found in adata")
    
    expression_matrix = adata[:, genes_in_adata].X.toarray()
    
    lr_expr_df = pd.DataFrame(
        expression_matrix.T,  
        index=genes_in_adata,  
        columns=adata.obs[label].values
    )
    
    #print("...........exact corresponding ScRNA data done..............")

    valid_pairs = lr[
        (lr['ligand'].isin(genes_in_adata)) & 
        (lr['receptor'].isin(genes_in_adata))
    ].copy()

    valid_pairs['l-r'] = valid_pairs['ligand'] + '-' + valid_pairs['receptor']
    lr_expr_pairs = valid_pairs.drop_duplicates(subset=["l-r"], keep="first")
    
    #print("...........generate lr_expr_pairs done..............")
    
    return lr_expr_df, lr_expr_pairs

def filter_scRNAdata_ltf(lr, adata, label):

    unique_genes = sorted(set(lr['ligand'].unique()) | set(lr['TF'].unique()))
    
    genes_in_adata = [gene for gene in unique_genes if gene in adata.var_names]
    if len(genes_in_adata) < len(unique_genes):
        print(f"Warning: {len(unique_genes) - len(genes_in_adata)} genes not found in adata")
    
    expression_matrix = adata[:, genes_in_adata].X.toarray()
    
    lr_expr_df = pd.DataFrame(
        expression_matrix.T,  
        index=genes_in_adata,  
        columns=adata.obs[label].values
    )
    
    #print("...........exact corresponding ScRNA data done..............")

    valid_pairs = lr[
        (lr['ligand'].isin(genes_in_adata)) & 
        (lr['TF'].isin(genes_in_adata))
    ].copy()

    valid_pairs['l-TF'] = valid_pairs['ligand'] + '-' + valid_pairs['TF']
    lr_expr_pairs = valid_pairs.drop_duplicates(subset=["l-TF"], keep="first")
    
    #print("...........generate lr_expr_pairs done..............")
    
    return lr_expr_df, lr_expr_pairs

def filter_scRNAdata_LRTF(lrtf, adata, label):
    sc_lr_expr_df, sc_lr_expr_pairs = filter_scRNAdata(lrtf[['ligand','receptor']], adata, label = label)
    sc_ltf_expr_df, sc_ltf_expr_pairs = filter_scRNAdata_ltf(lrtf[['ligand','TF']], adata, label = label)
    return sc_lr_expr_df, sc_lr_expr_pairs, sc_ltf_expr_df, sc_ltf_expr_pairs

def filter_stRNAdata(lr, adata):
    """
    Process spatial transcriptomics data to extract corresponding LR pairs and genes based on effective LRIs from bulk data.

    Parameters:
    lr (pd.DataFrame): Phenotype-specific LRI list.
    adata (AnnData): Preprocessed spatial transcriptomics data, recommended in TPM format.

    Returns:
    lr_expr_df (pd.DataFrame): Expression matrix of corresponding LR genes in spatial transcriptomics data.
    lr_expr_pairs (pd.DataFrame): List of corresponding LR pairs in spatial transcriptomics data.
    """
    unique_genes = sorted(set(lr['ligand'].unique()) | set(lr['receptor'].unique()))

    genes_in_adata = [gene for gene in unique_genes if gene in adata.var_names]
    if len(genes_in_adata) < len(unique_genes):
        print(f"Warning: {len(unique_genes) - len(genes_in_adata)} genes not found in adata")

    expression_matrix = adata[:, genes_in_adata].X.toarray()

    lr_expr_df = pd.DataFrame(
        expression_matrix.T,  
        index=genes_in_adata, 
        columns=adata.obs.index.tolist()
    )
    
    #print("...........exact corresponding ScRNA data done..............")

    valid_pairs = lr[
        (lr['ligand'].isin(genes_in_adata)) & 
        (lr['receptor'].isin(genes_in_adata))
    ].copy()

    valid_pairs['l-r'] = valid_pairs['ligand'] + '-' + valid_pairs['receptor']
    lr_expr_pairs = valid_pairs.drop_duplicates(subset=["l-r"], keep="first")
    
    #print("...........generate lr_expr_pairs done..............")
    
    return lr_expr_df, lr_expr_pairs

def filter_stRNAdata_ltf(lr, adata):

    unique_genes = sorted(set(lr['ligand'].unique()) | set(lr['TF'].unique()))

    genes_in_adata = [gene for gene in unique_genes if gene in adata.var_names]
    if len(genes_in_adata) < len(unique_genes):
        print(f"Warning: {len(unique_genes) - len(genes_in_adata)} genes not found in adata")

    expression_matrix = adata[:, genes_in_adata].X.toarray()

    lr_expr_df = pd.DataFrame(
        expression_matrix.T,  
        index=genes_in_adata, 
        columns=adata.obs.index.tolist()
    )
    
    #print("...........exact corresponding ScRNA data done..............")

    valid_pairs = lr[
        (lr['ligand'].isin(genes_in_adata)) & 
        (lr['TF'].isin(genes_in_adata))
    ].copy()

    valid_pairs['l-TF'] = valid_pairs['ligand'] + '-' + valid_pairs['TF']
    lr_expr_pairs = valid_pairs.drop_duplicates(subset=["l-TF"], keep="first")
    
    #print("...........generate lr_expr_pairs done..............")
    
    return lr_expr_df, lr_expr_pairs

def filter_stRNAdata_LRTF(lrtf, adata):
    sc_lr_expr_df, sc_lr_expr_pairs = filter_stRNAdata(lrtf[['ligand','receptor']], adata)
    sc_ltf_expr_df, sc_ltf_expr_pairs = filter_stRNAdata_ltf(lrtf[['ligand','TF']], adata)
    return sc_lr_expr_df, sc_lr_expr_pairs, sc_ltf_expr_df, sc_ltf_expr_pairs

def max_min_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def km_curve(mat):
    # Convert input to numpy array
    mat = np.array(mat)
    mat = mat[:, :2]
    mat = mat[np.argsort(mat[:, 0])]  # Sort by time
    
    # Initialize survival rate
    unique_times = np.unique(mat[:, 0])
    survival_rate = 1.0
    num_current = len(mat)
    survival_rates = np.ones(len(mat))
    
    for time in unique_times:
        ind = mat[:, 0] == time
        censor = mat[ind, 1]
        sr = survival_rate * (1 - np.sum(censor) / num_current)
        survival_rates[ind] = sr
        survival_rate = sr
        num_current -= np.sum(ind)
    
    return np.column_stack((mat, survival_rates))

def completerank(mat, complete=False):
    mat = np.array(mat)
    mat = mat[:, :2]
    
    original_indices = np.arange(len(mat))
    
    sorted_indices = np.argsort(mat[:, 0])
    mat = mat[sorted_indices]
    
    mat_curve = km_curve(mat)
    mat_completerank = np.column_stack((mat_curve, np.zeros(len(mat))))
    
    vect = mat_completerank[:, 0]
    vecs = mat_completerank[:, 1]
    
    for i in range(len(mat)):
        tA = mat_completerank[i, 0]
        rA = mat_completerank[i, 2]
        sA = mat_completerank[i, 1]
        
        if sA == 1:
            tBgttA = mat_completerank[vect > tA, 2]
            tBletA_sBeq0 = mat_completerank[(vect <= tA) & (vecs == 0), 2]
            tBeqtA_sBeq1 = mat_completerank[(vect == tA) & (vecs == 1), 2]
            
            rank = (0 if len(tBgttA) == 0 else 1 * len(tBgttA)) + \
                   (0 if len(tBletA_sBeq0) == 0 else np.sum(rA / tBletA_sBeq0)) + \
                   (0 if len(tBeqtA_sBeq1) == 0 else 0.5 * len(tBeqtA_sBeq1))
        
        if sA == 0:
            tBgetA_sBeq0 = mat_completerank[(vect >= tA) & (vecs == 0), 2]
            tBgetA_sBeq1 = mat_completerank[(vect >= tA) & (vecs == 1), 2]
            tBlttA_sBeq0 = mat_completerank[(vect < tA) & (vecs == 0), 2]
            
            rank = (0 if len(tBgetA_sBeq0) == 0 else np.sum(1 - 0.5 * tBgetA_sBeq0 / rA)) + \
                   (0 if len(tBgetA_sBeq1) == 0 else np.sum(1 - tBgetA_sBeq1 / rA)) + \
                   (0 if len(tBlttA_sBeq0) == 0 else np.sum(0.5 * rA / tBlttA_sBeq0))
        
        mat_completerank[i, 3] = rank
    
    mat_completerank[:, 3] -= 0.5
    mat_completerank[:, 3] /= np.max(mat_completerank[:, 3])
    
    mat_completerank = mat_completerank[np.argsort(sorted_indices)]
    
    if not complete:
        mat_completerank = mat_completerank[:, [0, 1, 3]]
    
    return mat_completerank #more high score more dangerous

def getProgConstraint(clin_df, interest):
    """
    Process clinical data to generate a clinical phenotype constraint matrix, sort the expression matrix, 
    and return the position range of samples for the phenotype of interest.

    Parameters:
    clin_df (DataFrame): clinical data, with samples as rows and 'OS.time, OS' as columns.
    interest (str): Focus on good prognosis (0) or poor prognosis (1).

    Returns:
    S (np.ndarray): Clinical phenotype constraint matrix.
    """
    rank = max_min_normalize(completerank(clin_df[['OS.time','OS']])[:,2])
    clin_df['completerank'] = rank
    if interest == 0: #focus on good prognosis
        S=np.diag(rank)
    elif interest == 1:
        S=np.diag(1-rank)
    else:
        raise ValueError("input required 0 or 1")


    return S

def getLinearConstraint(clin_df, by):
    """
    Process linear clinical data to generate a clinical phenotype constraint matrix
   
    Parameters:
    clin_df (DataFrame): Clinical data, with samples as rows.
    by (str): Name of the column to be normalized.
    
    Returns:
    S (np.ndarray): Normalized matrix.
    """
    if by not in clin_df.columns:
        raise ValueError(f"Column {by} not found in clin_df")
    
    column_values = clin_df[by].values
    normalized_values = max_min_normalize(column_values)
    
    S = np.diag(normalized_values)
    
    return S

def getBinaryConstraint(clin_df, lr_expr_df, by=None, phenotype_of_interest=None):
    """
    Process clinical data to generate a clinical phenotype constraint matrix, sort the expression matrix, 
    and return the position range of samples for the phenotype of interest.

    Parameters:
    clin_df (pd.DataFrame): clinical data, with samples as rows and phenotypes as columns.
    lr_expr_df (pd.DataFrame): Filtered L-R expression matrix.
    by (str): Clinical phenotype type.
    phenotype_of_interest (str): Phenotype of interest.

    Returns:
    S (np.ndarray): Clinical phenotype constraint matrix.
    Y (np.ndarray): Logistic regression labels, which can then be used for inferring CCI
    lr_expr_df (pd.DataFrame): Sorted L-R expression matrix.
    stage_intervals (dict): Position ranges of samples for each phenotype stage in the sorted data (start index, end index).
    sorted_clin_df (pd.DataFrame): Sorted clinical data.
    """
    sorted_clin_df = clin_df.sort_values(by=by)
    binary_labels = sorted_clin_df[by].values
    binary_labels_vector = np.where(binary_labels == phenotype_of_interest, 0, 1)
    Y = np.where(binary_labels == phenotype_of_interest, 1, 0).reshape(-1, 1)
    S = np.diag(binary_labels_vector)

    lr_expr_df = lr_expr_df[sorted_clin_df.index]
    stage_intervals = {}
    phenotype_indices = np.where(sorted_clin_df[by] == phenotype_of_interest)[0]
    contphe_indices = np.where(sorted_clin_df[by] != phenotype_of_interest)[0]
    if len(phenotype_indices) > 0:
        #phenotype_interval = (phenotype_indices[0], phenotype_indices[-1])# Interval is left-closed and right-open
        stage_intervals['phenotype'] = (phenotype_indices[0], phenotype_indices[-1])
        stage_intervals['control'] = (contphe_indices[0], contphe_indices[-1])
    else:
        phenotype_interval = (None, None) # If no samples of the phenotype of interest are found
        stage_intervals['control'] = (contphe_indices[0], contphe_indices[-1])

    stage_intervals = dict(sorted(stage_intervals.items()))

    return S, Y, lr_expr_df, stage_intervals, sorted_clin_df

def getOrderedConstraint(clin_df, lr_expr_df, by=None):
    """
    Process clinical data to generate a clinical phenotype constraint matrix, sort the expression matrix, 
    and return the position range of samples for the phenotype of interest.

    Parameters:
    clin_df (pd.DataFrame): clinical df, with samples as rows and phenotypes as columns.
    lr_expr_df (pd.DataFrame): Filtered L-R expression matrix.
    by (str): interesting clinical phenotype type (The inner element must be of type int, starting with 1).

    Returns:
    S (np.ndarray): Clinical phenotype constraint matrix.
    ordered_labels (np.ndarray): Ordered logistic regression labels, which can then be used for inferring CCI
    lr_expr_df (pd.DataFrame): Sorted L-R expression matrix.
    stage_intervals (dict): Position ranges of samples for each phenotype stage in the sorted data (start index, end index).
    sorted_clin_df (pd.DataFrame): Sorted clinical data.
    """
    sorted_clin_df = clin_df.sort_values(by=by, ascending=True)

    min_stage = sorted_clin_df[by].min()  
    max_stage = sorted_clin_df[by].max()  

    normalized_stages = (sorted_clin_df[by] - min_stage) / (max_stage - min_stage)

    normalized_stages_array = normalized_stages.to_numpy()
    S=np.diag(1-normalized_stages_array)
    ordered_labels = sorted_clin_df[by].values.reshape(-1, 1)

    lr_expr_df = lr_expr_df[sorted_clin_df.index]

    stages = sorted_clin_df[by].values

    stage_intervals = {}

    current_stage = stages[0]
    start_index = 0

    for i in range(1, len(stages)):
        if stages[i] != current_stage:
            stage_intervals[current_stage] = (start_index, i - 1)
            current_stage = stages[i]
            start_index = i
    
    stage_intervals[current_stage] = (start_index, len(stages) - 1)
    stage_intervals = dict(sorted(stage_intervals.items()))

    return S, ordered_labels, lr_expr_df, stage_intervals, sorted_clin_df


def getCoxelement(clin_df):
    """
    Process clinical data to generate time and event information.

    Parameters:
    clin_df (dataframe): clinical data, with samples as rows and 'Os.time, OS' as columns.

    Returns:
    time (np.ndarray): Array of time values.
    event (np.ndarray): Array of event values.
    """
    time,event = clin_df[['OS.time']].values,clin_df[['OS']].values

    return time.squeeze(), event.squeeze()

def build_communication_matrix(lr_expr_pairs, lr_expr_df):
    """
    Build a 2D communication matrix by calculating the product of expression levels for each ligand-receptor pair.

    Parameters:
    lr_expr_pairs (pd.DataFrame): DataFrame containing ligand-receptor pairs, must have columns 'l-r', 'ligand', 'receptor'.
    lr_expr_df (pd.DataFrame): DataFrame containing expression levels of ligands and receptors, with genes as rows and samples as columns.

    Returns:
    comm_matrix2 (pd.DataFrame): 2D communication matrix, with ligand-receptor pairs as rows, samples as columns, and values as the product of ligand and receptor expression levels.
    """
    # Initialize the communication matrix
    comm_matrix2 = pd.DataFrame(index=lr_expr_pairs['l-r'], columns=lr_expr_df.columns)

    # Calculate the product of expression levels for each ligand-receptor pair
    for i, row in lr_expr_pairs.iterrows():
        ligand = row['ligand']
        receptor = row['receptor']
        product = lr_expr_df.loc[ligand] * lr_expr_df.loc[receptor]
        comm_matrix2.loc[row['l-r']] = np.sqrt(product)

    return comm_matrix2

"""
def build_LRTF_communication_matrix(refined_pathways: pd.DataFrame, bulk_rna: pd.DataFrame):

    para：
    ----------
    refined_pathways : pd.DataFrame contains ['ligand', 'receptor', 'TF'] columns
    bulk_rna : pd.DataFrame row:(gene symbols), column:sample

    return：
    ----------
    comm_matrix2 : pd.DataFrame

    filtered_pathways = refined_pathways

    comm_matrix2 = pd.DataFrame(index=filtered_pathways['l-r-tf'], columns=bulk_rna.columns, dtype=float)

    for i, row in filtered_pathways.iterrows():
        ligand, receptor, tf = row['ligand'], row['receptor'], row['TF']
        try:
            ligand_expr = bulk_rna.loc[ligand]
            receptor_expr = bulk_rna.loc[receptor]
            tf_expr = bulk_rna.loc[tf]
            geom_mean = (ligand_expr * receptor_expr * tf_expr) ** (1 / 3)
            #geom_mean = np.exp(np.mean(np.log1p([ligand_expr, receptor_expr, tf_expr]), axis=0))

            comm_matrix2.loc[row['l-r-tf']] = geom_mean

        except KeyError:
            continue

    return comm_matrix2
"""

def build_LRTF_communication_matrix(refined_pathways: pd.DataFrame, bulk_rna: pd.DataFrame):
    """
    Parameters
    ----------
    refined_pathways : pd.DataFrame
        Must contain columns: ['ligand', 'receptor', 'TF', 'l-r-tf']
    bulk_rna : pd.DataFrame
        Rows: gene symbols; Columns: samples (non-negative expression preferred)
    nonneg_clip : bool
        If True, clip ligand/receptor/TF to >= 0 before geometric means.
    dtype : str or numpy dtype
        Output dtype (e.g., 'float32' to save memory)

    Returns
    -------
    comm_matrix2 : pd.DataFrame
        L–R–TF communication activity based on the GEOMETRIC MEAN between
        lr_activity = sqrt(L * R) and TF expression:
            score = sqrt(lr_activity * TF)
                 = (L * R * TF**2) ** 1/4
    """
    filtered_pathways = refined_pathways.copy()

    comm_matrix2 = pd.DataFrame(index=filtered_pathways['l-r-tf'], 
                                columns=bulk_rna.columns, dtype=float)

    for i, row in filtered_pathways.iterrows():
        ligand, receptor, tf = row['ligand'], row['receptor'], row['TF']
        try:
            ligand_expr = bulk_rna.loc[ligand]
            receptor_expr = bulk_rna.loc[receptor]
            tf_expr = bulk_rna.loc[tf]

            lr_activity = np.sqrt(ligand_expr * receptor_expr)

            eps = 1e-9
            #score = 2 / ((1 / (lr_activity + eps)) + (1 / (tf_expr + eps)))
            score = np.sqrt(lr_activity * tf_expr)

            comm_matrix2.loc[row['l-r-tf']] = score.values

        except KeyError:
            continue

    return comm_matrix2
