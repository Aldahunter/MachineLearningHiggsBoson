"""A Random Forest Classifier."""
import ctypes
import numpy as np
import multiprocessing as mp
from multiprocessing import sharedctypes as shared
import trees.functions as TF
import trees._BaseClasses as _BC
import trees._RandomForest._RandomForestFuncs as _RFF
from trees._BinaryDecisionTree \
    import _BinaryTreeEstimator as BDTE, _BinaryTreeClassifier as BDTC


class _RandomForestEstimator(_BC.BaseClassifier):
    """hyparams: - max_depth=None,
                 - min_samples_split=None,
                 - min_samples_leaf=None,
                 - max_observables=True-->[sqrt(n_observables)],
                 - n_trees=200,
                 - bs_frac=1.0,
                 - voting='soft',
                 - replacement=True."""
    _hparams = {'max_observables':True, 'n_trees':200, 'bs_frac':1.0,
                'voting':'soft', 'replacement':True}
    
    
    def _fit(self, observables, labels, impurity_fn=TF.gini_impurity,
             inform=False, processes=-1, timeout=True, OOBtest=False,
             **hyparams):
        
        # Update Hyperparamters
        self._hparams.update(hyparams)
        
        if self._hparams['max_observables'] is True:
            n_observables = np.sqrt(observables.shape[1])
            self._hparams['max_observables'] = int(round(n_observables))
        
        # Get list bootstrapped event indices for training trees
        indices = _RFF.bootstrap_indices(labels, self._hparams, OOBtest)
        
        # Decide whether to multiprocess fittings
        if (((self._hparams['n_trees']<=5) and (len(observables)<=10000))
        or (processes==1)):
            self._singleprocessFit(observables, labels,
                                   inform, impurity_fn, indices)
        else:
            self._multiprocessFit(observables, labels, inform, impurity_fn,
                                  indices, timeout, processes)
        
        # Update hyperparameters with tree's hyperparameters
        self._hparams.update({k:v for k,v in
                              self._forest[-1].get_hyparams().items()
                              if k not in self._hparams})
        self._fitted = True
            
    def _multiprocessFit(self, observables, labels, inform, impurity_fn,
                         indices, timeout, processes):
        
        # If timeout is True calculate the fitting time
        if timeout is True:
            timeout=self._trainingtime(self._hparams['n_trees'],observables)
            timeout *= 1.5
            print(f'Timeout: {timeout}')
        
        # Get number of processes
        processes = mp.cpu_count() if (processes < 0) else processes
        
        # Train trees on multiple processes
        self._forest = _RFF.multiprocessing_fit(observables, labels,
                                                indices, impurity_fn,
                                                processes, inform,
                                                timeout, self._hparams)
        
    def _singleprocessFit(self, observables, labels, inform, impurity_fn,
                          indices):
        
        # Iterate over list of bootstrapped event indices
        self._forest = []
        BinaryTree = BDTE if (self._hparams['voting']=="soft") else BDTC
        for n, sample_indices in enumerate(indices):
            
            # Create a Binary Decision Tree and fit to events
            tree = BinaryTree(**self._hparams)
            tree.fit(observables[sample_indices[0]],
                     labels[sample_indices[0]], impurity_fn=impurity_fn)
            self._forest.append(tree)
            
            # Inform progress
            if inform:
                print(f"#{n+1:3d} trees planted out of " + 
                      f"{self._hparams['n_trees']:3d} " +
                      f"({100*(n+1)/self._hparams['n_trees']:.2f}%)")

        
    def _predictions(self, observables, label=None, processes=-1,
                     timeout=60, *args, **kwargs):
        
        # Decide whether to multiprocess fittings
        if (((self._hparams['n_trees']<=5) and (len(observables)<=10000))
        or (processes==1)):
            return self. _singleprocessPredict(observables)
        else:
            return self._multiprocessPredict(observables, timeout,
                                             processes)
        
    def _multiprocessPredict(self, observables, timeout, processes):
        
        # If timeout is True calculate the fitting time
        if timeout is True:
            timeout=self._trainingtime(self._hparams['n_trees'],observables)
            timeout *= 1.1
        
        # Get number of processes
        processes = mp.cpu_count() if (processes < 0) else processes
        
        # Precit events on each tree on multiple processes
        predictions = _RFF.multiprocessing_predict(self._forest,
                                                   observables,
                                                   processes, timeout)
        
        # Return mean of each event
        return np.mean(np.c_[predictions], axis=0)
        
    def _singleprocessPredict(self, observables):
        
        # Iterate over each tree
        predictions = []
        for tree in self._forest:
            
            # Create predict events for this tree
            tree.predict(observables)
            predictions.append(tree)
        
        # Return mean of each event
        return np.mean(np.c_[predictions], axis=0)
    
    
    def get_depths(self):
        return [tree.get_depth() for tree in self._forest]             
    def get_min_depth(self):
        return min(self.get_depths())
    def get_max_depth(self):
        return max(self.get_depths())
    def get_range_depths(self):
        depths = self.get_depths()
        return (min(depths), max(depths))
    
#     def get_probabilities(self):
#         probabilities = TF.leaf_probabilities(self.tree)
#         return sorted(probabilities, key=lambda e: (e is not None, e))
    
#     def get_path(self, probability):
#         root_node = TF.get_probability_path(self.tree, probability)
#         return TN.TreePath(root_node, probability)
    
#     def get_num_leaves(self):
#         return TF.get_num_leaves(self.tree)
    
    def contains_None(self):
        nones = [tree.contains_None() for tree in self._forest]
        return (True in nones)
    
    
    @property
    def forest(self): return self._forest
    @property
    def size(self): return self._hparams['n_trees']
    @property
    def n_trees(self): return self._hparams['n_trees']
    
    
    @staticmethod
    def _trainingtime(n_trees, observables):
        A, a, b = 3.20525123e-07, 1.03130986e+00, 1.38976666e+00
        c, d, e =  1.57069341e+00, .08743505e+00, 6.13200046e-03
        n_events, n_features = observables.shape
        
        return A * (a*n_trees * b*n_features
                    * c*n_events * d*np.log(e*n_events)) + 1
    
    
    def __repr__(self):
        return (f"RandomForest(Fitted: {self._fitted}; "
                + self._hparams.__repr__()[1:-1] + ")")
    __str__ = __repr__
    
    def __getitem__(self, key):
        return self._hparams[key]
    
    def __iter__(self):
        return iter(self._forest)

    
class _RandomForestClassifier(_RandomForestEstimator, _BC.BaseClassifier):
    __sclass__ = "RandomForestClassifier"
    

class _ExtraTreesEstimator(_RandomForestEstimator):
    __sclass__ = "ExtraTreesEstimator"
    _hparams = {'max_observables':True, 'n_trees':200, 'bs_frac':1.0,
                'voting':'soft', 'replacement':True, 'random_minimize':1.0}

class _ExtraTreesClassifier(_RandomForestClassifier):
    __sclass__ = "ExtraTreesClassifier"
    _hparams = {'max_observables':True, 'n_trees':200, 'bs_frac':1.0,
                'voting':'soft', 'replacement':True, 'random_minimize':1.0}