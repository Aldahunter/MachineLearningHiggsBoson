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
    np.random.shuffle(bkg_indices)

    # Compile the indices
    train_indices = np.r_[(sgn_indices[:n_sgn], bkg_indices[:n_bkg])]
    valid_indices = np.r_[(sgn_indices[n_sgn:], bkg_indices[n_bkg:])]

    # Return the indices
    return train_indices, valid_indices

def _threshold(labels, probabilities):
    return AME.find_p_threshold(labels, probabilities)

def sanity_checks(observables, labels, sample_weights, normalise_weights):
    
    # Ensure observables and labels match-up.
    n_events = len(labels)
    if not len(observables) == n_events:
        raise ValueError("Observables and labels must have the same "
                         "first dimension. You have given "
                         f"{len(observables)} and {n_events}.")

    # If sample_weights not given give each events a weight of one.
    if sample_weights is None:
        sample_weights = np.full(n_events, 1.0/n_events, dtype=np.float64)
    else:
        if len(sample_weights) != n_events:
            raise ValueError("'Sample_weights' must have the same "
                             "first dimension as 'observables' and "
                             "'labels'. You have given "
                             f"{len(sample_weights)}.")
        # Normalise sample weights
        if normalise_weights:
            sample_weights = np.divide(sample_weights, sample_weights.sum(),
                                       dtype=np.float64)

    # Ensure arrays are in the correct datatypes and memory storage
    observables = np.asfortranarray(observables)
    labels = np.ascontiguousarray(labels, dtype=np.bool)
    sample_weights = np.ascontiguousarray(sample_weights,
                                          dtype=np.float64)
    
    # Return corrected arrays
    return observables, labels, sample_weights