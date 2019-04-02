"""A Module to hold the functions associated with multithreading of classifiers."""

import numpy as np
import trees.functions as TF
from trees.functions import gini_impurity
from trees._classifiers import _BinaryDecisionTree as BinaryDecisionTree


### Functions ###
def multiprocessing_initFit(n_obs, RawObs, RawLabs):
    global sharedLabels
    global sharedObservables
    
    sharedLabels = np.frombuffer(RawLabs)
    sharedObservables = np.frombuffer(RawObs).reshape((len(sharedLabels),
                                                       n_obs.value))

def multiprocessing_fit(params):
    n, sample_indices, impurity_fn, hparams, inform = params
    
    tree = BinaryDecisionTree(**hparams)
    tree.fit(sharedObservables[sample_indices],
             sharedLabels[sample_indices],
             impurity_fn=impurity_fn)
    
    if inform:
        print(f"Tree #{n+1:3d} planted out of {hparams['n_trees']:3d}")
              
    return tree

def multiprocessing_initPredict(RawArr, i, j):
    global sharedObservables
              
    sharedObservables = np.frombuffer(RawArr).reshape((i.value, j.value))

def multiprocessing_predict(classifier):
    return classifier.predict(sharedObservables)