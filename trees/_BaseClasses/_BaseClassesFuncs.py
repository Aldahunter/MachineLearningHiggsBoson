import numpy as np
import analysis.metrics as AME

def _test_validate_split(labels, train_val_frac):
    
    # Get signal and background indices
    sgn_indices = np.where(labels != 0)[0]
    bkg_indices = np.where(labels == 0)[0]
    
    # Get signal and background sample sizes
    n_sgn = round(len(sgn_indices) * train_val_frac)
    n_bkg = round(len(bkg_indices) * train_val_frac)
    
    # Randomly shuffle indices
    np.random.shuffle(sgn_indices)
    np.random.shuffle(bkg_indices);

    # Compile the indices
    train_indices = np.r_[(sgn_indices[:n_sgn], bkg_indices[:n_bkg])]
    valid_indices = np.r_[(sgn_indices[n_sgn:], bkg_indices[n_bkg:])]

    # Return the indices
    return train_indices, valid_indices

def _threshold(labels, probabilities):
    return AME.find_p_threshold(labels, probabilities)