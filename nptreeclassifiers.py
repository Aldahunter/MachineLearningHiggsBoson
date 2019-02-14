### Imports ###
import numpy as np

import treefunctions as TF



### Functions ###
df_to_ML_input = TF.df_to_ML_input



### Single Learners ###
class Estimator(object):
    
    def __init__(self, **hyparams):
        self._hparams = hyparams
    
    
    def set_params(self, **params):
        for key, param in params.items():
            self._hparams[key] = param
            
    
    def get_params(self, deep=True):
        return self._hparams 

    

class BinaryTreeClassifier(Estimator):
    
    def __init__(self, **hyparams):
        """kwargs:
            - max_depth=None,
            - min_samples_split=None,
            - min_samples_leaf=None,
            - min_weight_fraction_leaf=None,
            - max_leaf_nodes=None,
            - max_features=None"""
        self._hparams = hyparams
    
    
    def plant_tree(self, observables, labels, impurity_fn=TF.gini_impurity,
                   **kwargs):
        self._hparams.update(kwargs)
        self._tree = TF.grow_tree(observables, labels, impurity_fn,
                                  **self._hparams)
    fit = plant_tree
    
    
    def show_tree(self, spacing=' ', trimmed=True, rounded=True):
        TF.show_tree(self.tree, 'Root', spacing, trimmed, rounded)
    
    
    def classify(self, observables, label=None):
        tree = self.tree
        return np.fromiter(( TF.tree_classify(datum, tree)
                             for datum in observables ),
                           float, len(observables))
    predict = classify
    
    
    def get_depth(self):
        return TF.tree_depth(self.tree)
    
    
    @property
    def tree(self): return self._tree



### Ensemble Learners ###