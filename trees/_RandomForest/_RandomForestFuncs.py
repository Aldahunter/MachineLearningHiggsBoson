"""A module to hold the functions associated with a Random Forest \
classifier."""

import ctypes
import numpy as np
import trees.functions as TF
import multiprocessing as mp
from multiprocessing \
    import sharedctypes as shared
from trees._BinaryDecisionTree \
    import _BinaryTreeEstimator as BinaryTreeEstimator, \
           _BinaryTreeClassifier as BinaryTreeClassifier


### Functions ###
def bootstrap_indices(labels, hyparams, OOBtest=False):
        
        # Get signal and background indices
        sgn_indices = np.where(labels != 0)[0]
        bkg_indices = np.where(labels == 0)[0]
        
        # Get signal and background sample sizes
        n_sgn = round(len(sgn_indices) * hyparams['bs_frac'])
        n_bkg = round(len(bkg_indices) * hyparams['bs_frac'])
        
        # Iterate over each tree and generate indices
        for _ in range(hyparams['n_trees']):
            
            if hyparams['replacement'] == True:
                # Randomly chose indices from possible set
                r_sgn_indices = np.random.choice(sgn_indices, size=n_sgn)
                r_bkg_indices = np.random.choice(bkg_indices, size=n_bkg)
                
            else:
                # Randomly shuffle indices
                np.random.shuffle(sgn_indices)
                np.random.shuffle(bkg_indices)
                # Take the first n_xxx indices
                r_sgn_indices = sgn_indices[:n_sgn]
                r_bkg_indices = bkg_indices[:n_bkg]

            # Combine and sort into train indices
            train_indices = np.sort(np.concatenate((r_sgn_indices,
                                                    r_bkg_indices)))
            
            # If no Out-Of-the-Bag testing, just return train indices
            if not OOBtest:
                yield (train_indices,)
                continue
            
            if hyparams['replacement'] == True:
                # Find the set difference
                r_sgn_indices = np.setdiff1d(sgn_indices, r_sgn_indices)
                r_bkg_indices = np.setdiff1d(bkg_indices, r_bkg_indices)
                
            else:
                # Take the last n_xxx indices
                r_sgn_indices = sgn_indices[n_sgn:]
                r_bkg_indices = bkg_indices[n_bkg:]
            
            # Sort into train indices
            test_indices = np.sort(np.concatenate((r_sgn_indices,
                                                   r_bkg_indices)))
            
            # Generate train and test indices
            yield (train_indices, test_indices)


def _multiprocessing_initFit(n_obs, RawObs, RawLabs):
    global sharedLabels
    global sharedObservables
    sharedLabels = np.frombuffer(RawLabs)
    sharedObservables = np.frombuffer(RawObs).reshape((len(sharedLabels),
                                                       n_obs.value))

def _multiprocessing_sfit(params):  # Soft Voting
    n, sample_indices, impurity_fn, hparams, inform = params
    
    tree = BinaryTreeEstimator(**hparams)
    tree.fit(sharedObservables[sample_indices],
             sharedLabels[sample_indices],
             impurity_fn=impurity_fn)
    
    if inform:
        print(f"Tree #{n+1:3d} planted out of {hparams['n_trees']:3d}")
    return tree
def _multiprocessing_hfit(params):  # Hard Voting
    n, sample_indices, impurity_fn, hparams, inform = params
    
    tree = BinaryTreeClassifier(**hparams)
    tree.fit(sharedObservables[sample_indices],
             sharedLabels[sample_indices],
             impurity_fn=impurity_fn)
    
    if inform:
        print(f"Tree #{n+1:3d} planted out of {hparams['n_trees']:3d}")
    return tree

def multiprocessing_fit(observables, labels, indices_list, impurity_fn,
                        processes, inform, timeout, hparams):
             
    n_obs = shared.RawValue(ctypes.c_uint, observables.shape[-1])
    RawObs = shared.RawArray(ctypes.c_double, observables.flatten())
    RawLabs = shared.RawArray(ctypes.c_double, labels)
    RawArr_info = (n_obs, RawObs, RawLabs)

    predict_kwargs = [(n, sample_indices[0], impurity_fn, hparams, inform)
                      for n, sample_indices in enumerate(indices_list)]
    
    pool_kwargs = {'processes': processes, 'initargs': RawArr_info,
                   'initializer': _multiprocessing_initFit}        
    
    with mp.Pool(**pool_kwargs) as pool:
        if hparams['voting'] == "soft":
            results = pool.map_async(_multiprocessing_sfit, predict_kwargs)
        else:
            results = pool.map_async(_multiprocessing_hfit, predict_kwargs)
        results.get(timeout)
    return results._value

              
def _multiprocessing_initPredict(RawObs, i, j):
    global sharedObservables
    sharedObservables = np.frombuffer(RawObs).reshape((i.value, j.value))

def _multiprocessing_predict(DecisionTree):
    return DecisionTree.predict(sharedObservables)

def multiprocessing_predict(forest, observables, processes, timeout):
    shape = observables.shape
    i = shared.RawValue(ctypes.c_uint, shape[0])
    j = shared.RawValue(ctypes.c_uint, shape[1])
    RawObs = shared.RawArray(ctypes.c_double, observables.flatten())
    RawArr_info = (RawObs, i, j)

    pool_kwargs = {'processes': processes, 'initargs': RawArr_info,
                   'initializer': _multiprocessing_initPredict}
    with mp.Pool(**pool_kwargs) as pool:
        results =  pool.map_async(_multiprocessing_predict, forest)
        results.get(timeout)

    return results._value