from glob import glob
import pandas as pd

from sklearn.model_selection import train_test_split



def train_val_files(s1_dir, seed, mode=None):
    imgs, lbls = [],[]
    
    S1_paths = sorted(glob(s1_dir))[:900]       #900
    train_s1, val_s1 = train_test_split(S1_paths,
                                        test_size=0.15,
                                        random_state=seed,
                                        shuffle=True)
    
    # Get the corresponding labels for S1
    train_lbl_s1 = [path.replace('c2smsfloods_v1_source_s1',
                                 'c2smsfloods_v1_labels_s1_water') for path in train_s1]
    val_lbl_s1   = [path.replace('c2smsfloods_v1_source_s1',
                                 'c2smsfloods_v1_labels_s1_water') for path in val_s1]

    
    # Get the corresponding S2 images 
    train_s2 = [path.replace('s1','s2') for path in train_s1]
    
    train_lbl_s2 = [path.replace('s1','s2') for path in train_lbl_s1]
    
    
    if mode == 'train':
        for S1_path, S1_lbl_path in zip(train_s1, train_lbl_s1):
            img = glob(S1_path+'/*.png')[0]
            lbl = glob(S1_lbl_path+'/*.tif')[0]
            imgs.append(img)
            lbls.append(lbl)

        df = pd.DataFrame({'source': imgs, 'label':lbls})
    
    else:
        for S1_path, S1_lbl_path in zip(val_s1, val_lbl_s1):
            img = glob(S1_path+'/*.png')[0]
            lbl = glob(S1_lbl_path+'/*.tif')[0]
            imgs.append(img)
            lbls.append(lbl)
        
        df = pd.DataFrame({'source': imgs, 'label':lbls}) 

    return df