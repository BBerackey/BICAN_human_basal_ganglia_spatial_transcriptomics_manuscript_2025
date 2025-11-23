import anndata as ad
import os 
from multiprocessing import Pool
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm 
from scipy import ndimage
import statsmodels.api as statsmodels

def run(celltype):
    print(f'Started analysis for {celltype}......')
    # Load the data
    MERFISH_adata = ad.read_h5ad(r'./processed_data/2724_Gpe_Gpi_leiden_clustered_mapmycellAnnotated.h5ad')
    spatialModuleAnnotation = pd.read_csv(r'./processed_data/AP26a_MERFISH/GroupLevelspatialModule_MatrixStriosome_included.csv', index_col=0)
    MERFISH_adata.obs = MERFISH_adata.obs.join(spatialModuleAnnotation['GrayMatter_MatrixStriosome_clustered'])
    # get the distance from Striosome [as per the result from striosome celltype distribution analysis]
    Striosome_distance = pd.read_csv(r'./processed_data/AP26a_MERFISH/Striosome_distance.csv',index_col = 0)
    MERFISH_adata.obs = MERFISH_adata.obs.join(Striosome_distance)
    # Normalizze the count matrix
    MERFISH_adata.X = MERFISH_adata.layers['X_raw'].copy()
    total_count = MERFISH_adata.X.sum(axis=-1)
    sc.pp.normalize_total(MERFISH_adata, target_sum = np.median(total_count))
    sc.pp.log1p(MERFISH_adata)
    # To save memmoery copy the normalized expression matrix and delete the data 
    expN = MERFISH_adata.X.copy()
    genes = np.array(MERFISH_adata.var_names)
    Group_name = np.array(MERFISH_adata.obs['Group_name'].values)
    Strio_distance = MERFISH_adata.obs['striosome_distance'].values
    del MERFISH_adata

    # +++++++++++++++++++++++++ Gene expression gradient per cell type +++++++++++++++++++++++
    num_permutation = 10000
    dm,dM = -500,500
    resc = 50
    cells_in_range = (Strio_distance > dm) & (Strio_distance < dM)

    # 1. Compute the mean expression within each binned_distance
    cellType_cond = np.isin(Group_name,celltype)
    Strio_distance_bin = (np.round(Strio_distance[cells_in_range & cellType_cond]/resc)*resc).astype(int) # digitize the coordinate by factor of 50 units
    range_ = np.unique(Strio_distance_bin)
    
    # 2. For each gene, estiamate the expression explained by the distance from striosome boarder
    final_results = {celltype:{}}
    for gn in tqdm(genes, desc =f'computing Gene gradient for celltype: {celltype}'):
        # 2.1. Get the mean expression within the digitized coordiantes
        keep = genes == gn
        Xpr = expN[:,keep].squeeze()
        Xpr_bin_gn = ndimage.mean(Xpr[cells_in_range & cellType_cond],Strio_distance_bin,range_)

        # 2.2. Build the null distribution based on the explained variance by the distance from the Striosome
        null_distr = []
        for permute_idx in range(num_permutation):
            Xpr_bin_gn_permuted = np.random.permutation(Xpr_bin_gn) # shuffle the data
            # Compute the lowess regression to get the value estimated by distance 
            res = statsmodels.nonparametric.lowess(Xpr_bin_gn_permuted,range_,frac = 0.8, return_sorted = False)
            # Compute the explained variance by the distance from striosomes
            unexplained_var = np.var(Xpr_bin_gn_permuted - res)
            total_var = np.var(Xpr_bin_gn_permuted)
            explained_var = 1- (unexplained_var/total_var)
            null_distr.append(explained_var)
        # 2.3 Get the statistics based on the null distribution 
        null_distr = np.array(null_distr)
        # fit the actual data
        
        res_observed = statsmodels.nonparametric.lowess(Xpr_bin_gn,range_,frac = 0.8, return_sorted = False)
        unexplained_var = np.var(Xpr_bin_gn - res_observed)
        total_var = np.var(Xpr_bin_gn)
        explained_var_observed = 1- (unexplained_var/total_var)

        expected_mu = np.mean(null_distr)
        expected_std = np.std(null_distr)

        zscore = (explained_var_observed - expected_mu)/(expected_std + 1e-11)# how many standard deviations the observation is from the mean
        fold_change = explained_var_observed/(expected_mu + 1e-11) # by how much the observed change is greater than the mean
        p_value = (1 + (null_distr >= explained_var_observed).sum())/(1 + len(null_distr)) # Measures the change that a random permutation gives a value at least as extreme as the observed
        final_results[celltype][gn] = {'zscore':zscore,'fold_change':fold_change,'p_value':p_value,
                                       'Xpr_bin_est':res_observed,'Xpr_bin_gn':Xpr_bin_gn}
    
    # save the result 
    np.save(r'./processed_data/AP26a_MERFISH'+ os.sep + '_'.join(celltype.split(' '))+'_GenegradAnalysis.npy', final_results,allow_pickle = True)


if __name__ == "__main__":
    
    if True:
        with Pool(processes=3) as pool:
            print('starting pool')
            result = pool.map(run,['Astrocyte', 'Endo', 'ImOligo', 'Oligo OPALIN', 'Oligo PLEKHG1',
            'STRd D1 Matrix MSN', 'STRd D1 Striosome MSN',
            'STRd D2 Matrix MSN', 'STRd D2 Striosome MSN'])


