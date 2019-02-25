### Imports ###
import numpy as np

import dwrangling.dataframes as DWDF
import trees.functions as TF
import trees.nodes as TN



### Single Learners ###
class Estimator(object):
    
    def __init__(self, **hyparams):
        self._hparams = hyparams
    
    
    def set_params(self, **hparams):
        self._hparams.update(hparams)
    set_hyparams = set_params
            
    
    def get_params(self, deep=True):
        return self._hparams
    get_hyparams = get_params

    

class BinaryTreeClassifier(Estimator):
    
    def __init__(self, **hyparams):
        """hyparams:
            - max_depth=None,
            - min_samples_split=None,
            - min_samples_leaf=None,
            - min_weight_fraction_leaf=None,
            - max_leaf_nodes=None,
            - max_features=None"""
        self._hparams = hyparams
        self._fitted = False
    
    
    def plant(self, observables, labels, impurity_fn=TF.gini_impurity,
                   **hyparams):
        self._hparams.update(hyparams)
        self._tree = TF.grow_tree(observables, labels, impurity_fn,
                                  **self._hparams)
        self._fitted = True
    fit = plant
    
    
    def show(self, spacing=' ', trimmed=True, rounded=True):
        TF.show_tree(self.tree, 'Root', spacing, trimmed, rounded)
    
    
    def classify(self, observables, label=None):
        return np.fromiter(( TF.tree_classify(datum, self.tree)
                             for datum in observables ),
                           float, len(observables))
    predict = classify
    
    
    def get_depth(self):
        return TF.tree_depth(self.tree)
    
    def get_probabilities(self):
        return TF.leaf_probabilities(self.tree)
    
    def get_path(self, probability):
        root_node = TF.get_probability_path(self.tree, probability)
        return TN.TreePath(root_node, probability)
    
    def get_num_leaves(self):
        return TF.get_num_leaves(self.tree)
    
    def contains_None(self):
        return TF.tree_contain_None(self.tree)
    
    
    @property
    def tree(self): return self._tree
    
    
    def __repr__(self):
        return (f"BinaryTreeClassifier(Fitted: {self._fitted}; "
                + self._hparams.__repr__()[1:-1] + ")")
    __str__ = __repr__
    
    # Define how classifier is indexed.
    def __getitem__(self, key):
        
        # If indexed, return indexed root node.
        return self.tree[key]
        



### Ensemble Learners ###