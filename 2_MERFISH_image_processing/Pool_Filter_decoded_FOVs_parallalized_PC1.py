import sys
from ioMicroHumanBG_version2 import *
import numpy as np
import napari
import os
from multiprocessing import Pool, set_start_method

def get_score(dec):
    H = np.nanmedian(dec.XH_pruned[...,-3],axis=1)
    n1bits = dec.XH_pruned.shape[1]
    from itertools import combinations
    combs = np.array(list(combinations(np.arange(n1bits),2)))##chose 2 out the 4 on bits as a combination
    X = dec.XH_pruned[:,:,:3] #zxy position
    D = np.nanmean(np.linalg.norm(X[:,combs][:,:,0]-X[:,combs][:,:,1],axis=-1),axis=1)#calculate norm(magnitude of xyz vector) for each combination, then mean, to generate average transcript spot position distance
    db = dec.dist_best
    score = np.array([H,-D,-db]).T #H-brightness, -D: zxy position 
    #score = np.sort(score,axis=1)
    #score_ref = np.sort(score,axis=0)
    dec.score = score
    return score
def set_scoreA(dec):
    score_ref = dec.score_ref
    score = dec.score
    from scipy.spatial import KDTree
    scoreA = np.zeros(len(score))
    for iS in range(score.shape[-1]):
        dist_,inds_ = KDTree(score_ref[:,[iS]]).query(score[:,[iS]])
        scoreA+=np.log((inds_+1))-np.log(len(score_ref))
    dec.scoreA = scoreA



# Run across the data

def filter_spots(fov,redo=False): 
    iset = fov.split('zscan')[-1].split('_')[0]
    if not iset:
        iset = '1' # Temporary fix, manual set iset to 1
    final_save_folderF = r'/mnt/Y/Lab/Raw_custom_MERFISH_Imaging_data/20250511_XXhBG_1311/HumanBG_MERFISH_Filtered_Spots'
    if not os.path.exists(final_save_folderF): os.makedirs(final_save_folderF)
    final_fl_save = final_save_folderF+os.sep+fov+'--matrix.npz'
    save_fl = final_save_folderF+os.sep+fov+'--spots.npz'
    print(save_fl)
    print(final_fl_save)
    dec=None
    if not os.path.exists(final_fl_save) or redo:
        save_folder = r'/mnt/Y/Lab/Raw_custom_MERFISH_Imaging_data/20250511_XXhBG_1311/MERmake_decoding'
        dec = decoder_simple(save_folder,fov,set_='')
        dec.ncols = 3
        dec.load_decoded()
        dec.dist_best=np.load(dec.decoded_fl)['dist_best']
        # keep = np.sum(dec.XH_pruned[:,:,-3]>5000,axis=-1)>=3
        # #keep&=dec.dist_best>0.25
        # dec.XH_pruned=dec.XH_pruned[keep]
        # dec.dist_best=dec.dist_best[keep]
        # dec.icodesN = dec.icodesN[keep]
        
        score = get_score(dec)
        scores_ref_fl = save_folder + os.sep + 'scores_reference.npy'
        if not os.path.exists(scores_ref_fl):
            
            score_ref = np.sort(dec.score,axis=0)
            dec.score_ref = score_ref
            np.save(scores_ref_fl,score_ref)
        else:
            dec.score_ref = np.load(scores_ref_fl)
        set_scoreA(dec)
        
        
        dec.th = -0.5
        keep = dec.scoreA>dec.th
        icodesN = dec.icodesN[keep]
        XHfpr = XH_pruned = dec.XH_pruned[keep]
        
        Xpr = np.nanmean(XH_pruned,axis=1)

        XHf = Xpr
        keepf=keep
        XF = XHf[:,[0,1,2,-5,-4,-3,-2,-1,-1,-1,-1]]
        #zc,xc,yc,bk-7,a-6,habs-5,hn-4,h-3
        XF[:,-1] = dec.scoreA[keepf]
        XF[:,-2] = np.where(keepf)[0]
        mnD = np.nanmean(np.linalg.norm((XHf[:,np.newaxis]-XHfpr)[:,:,:3],axis=-1),axis=-1)
        XF[:,-3]=mnD
        mnH = np.nanmean(np.abs((XHf[:,np.newaxis]-XHfpr)[:,:,-3]),axis=-1)
        XF[:,-4]=mnH
        genesf = dec.gns_names[icodesN]
  
        Xmol = Xpr[:,:3]
        
    
        ### load segmentation
        seg_fl = '/mnt/Y/Lab/Raw_custom_MERFISH_Imaging_data/20250511_XXhBG_1311/Segmentation' + os.sep + f'{fov}--H1_MER_set1--dapi_segm.npz'
        segm = np.load(seg_fl)['segm']
        shape = np.load(seg_fl)['shape']


        #  Correct drift to segmentation 
        # 1. set the Hyb ID used for segmentation 
        Hyb = os.path.basename(seg_fl).split('--')[1]
        # Load in the drift correction value 
        drifts,all_fld,_,_ = np.load('/mnt/Y/Lab/Raw_custom_MERFISH_Imaging_data/20250511_XXhBG_1311/MERmake_decoding' + os.sep + f'driftNew_{fov}--.pkl',allow_pickle=True) 
        all_Hybs = [os.path.basename(x) for x in all_fld]
        segm_Hyb_Idx = all_Hybs.index(Hyb)
        drift = drifts[segm_Hyb_Idx][0]

        Xmol= Xmol + drift
        XF[:,:3] = XF[:,:3] + drift ### bring to reference frame!! where segmentation is computed

        XR = Xmol*segm.shape/shape
        
        ### assign to cells

        segm_ = expand_segmentation(segm,10)
        XR = np.round(XR).astype(dtype=int)
        cell_id = np.zeros(len(XR),dtype=np.float32)
        keepR = np.all((XR>=0)&(XR<segm_.shape),axis=-1)
        cell_id[keepR] = segm_[tuple(XR[keepR].T)]
        
        ifov = int(dec.fov.split('_')[-1].split('.')[0])
        isets = np.array([iset]*len(cell_id))[:,np.newaxis]
        ifovs = np.array([ifov]*len(cell_id))[:,np.newaxis]
        cell_id = cell_id[:,np.newaxis]
        #XF_ = np.concatenate([XF[good],cell_id,ifovs,isets],axis=-1)
        
        XF_ = np.concatenate([XF.astype(np.float32),cell_id.astype(np.float32),ifovs,isets],axis=-1)
        XF_ = XF_[:,list(np.arange(XF_.shape[-1]))+[-1,-1]]
        # get the X and Y FOV coords from the stage metadata of the Hyb used for segmentation 
        FOV_meta = '/mnt/Y/Lab/Raw_custom_MERFISH_Imaging_data/20250511_XXhBG_1311'  + os.sep + all_Hybs[segm_Hyb_Idx] + os.sep + f'{fov}.xml'
        FOV_meta = open(FOV_meta).read()
        tag = '<stage_position type="custom">'
        xfov, yfov = eval(FOV_meta.split(tag)[-1].split('</')[0])
        XF_[:,-2:]=np.array([xfov,yfov])

        
        header = ['z','x','y','abs_brightness','cor','brightness','color','mean_bightness_variation','mean_distance_variation',
                  'index_from_XH_pruned','score','cell_id','ifov','iset','xfov','yfov']
        
        #segm_ = expand_segmentation(segm,10)
        dec.im_segm_ = segm_
        icells,vols = np.unique(dec.im_segm_,return_counts=True)
        cms = np.array(ndimage.center_of_mass(np.ones_like(dec.im_segm_),dec.im_segm_,icells))
        #icells,vols = np.unique(dec.im_segm_,return_counts=True)
        cms = np.array(ndimage.center_of_mass(np.ones_like(dec.im_segm_),dec.im_segm_,icells))
        cellinfo = cms[:,[0,0,0,1,2,0,0]]
        cellinfo[:,0]=icells
        cellinfo[:,1]=vols
        dec.xfov,dec.yfov = xfov, yfov
        cellinfo[:,-2:]=dec.xfov,dec.yfov
        header_cells = ['cell_id','volm','zc','xc','yc','xfov','yfov']
    
        np.savez_compressed(save_fl,XF=XF_.astype(np.float32),
                            genes = genesf,cellinfo=cellinfo.astype(np.float32),header_cells=header_cells,header=header)
        
        index_cells = cell_id.squeeze()
        genes= dec.gns_names
        igenes= np.arange(len(genes))
        cellsu = icells
        M = np.zeros([int(np.max(icells)+1),len(genes)],dtype=int)
        Mgn = int(np.max(igenes)+1)
        iVALS,iCTS = np.unique(index_cells.astype(int)*Mgn+icodesN,return_counts=True)
        M[iVALS//Mgn,iVALS%Mgn]=iCTS
        M=M[icells]
        
        np.savez_compressed(final_fl_save,Mmer=M,genes_mer = genes,cells_mer = [fov+'--'+str(e) for e in cellsu])
    return dec


if __name__ == '__main__':

    # Calibrate with FOV
    # Calcualte a refence score based on Randomly selected FOV
    if True:
        fov = 'Conv_zscan__0304'
        save_folder = r'/mnt/Y/Lab/Raw_custom_MERFISH_Imaging_data/20250511_XXhBG_1311/MERmake_decoding'
        dec = decoder_simple(save_folder,fov,set_='')
        dec.ncols = 3
        dec.load_decoded()
        dec.dist_best=np.load(dec.decoded_fl)['dist_best']

        score = get_score(dec)
        scores_ref_fl = save_folder + os.sep + 'scores_reference.npy'

        if not os.path.exists(scores_ref_fl):
            score_ref = np.sort(dec.score,axis=0)
            dec.score_ref = score_ref
            np.save(scores_ref_fl,score_ref)
        else:
            dec.score_ref = np.load(scores_ref_fl)
        set_scoreA(dec)

        # set a threshold 
        dec.th=-0.5

    # if False:
    FOVs_ToRun = np.load(r'/mnt/Y/Lab/Raw_custom_MERFISH_Imaging_data/20250511_XXhBG_1311/MERFISH_Analysis/pre_processing/FOV_to_filter.npy')
    fovs = [f'Conv_zscan__{x:04d}' for x in FOVs_ToRun]
  
  
    with Pool(processes=3) as pool:
        print('starting pool')
        result = pool.map(filter_spots,fovs)
