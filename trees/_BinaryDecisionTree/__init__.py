"""A library to hold a Binary Decision Tree classifier and its associated \
functions and classes."""
import numpy as np
import trees.functions as TF
import trees._BaseClasses as _BC
import trees._BinaryDecisionTree._BDTFuncs as _BDTF
import trees._BinaryDecisionTree._BDTClasses as _BDTC


class _BinaryTreeEstimator(_BC.BaseEstimator):
    """hyparams: - max_depth=12,
                 - min_samples_leaf=0,
                 - min_samples_split=800,
                 - max_observables=None,
                 - min_weight_fraction_leaf=None"""
    __sclass__ = "BinaryTreeEstimator"
    _hparams = {'max_depth':12,
                'min_samples_leaf':0,
                'min_samples_split':800,
                'max_observables':None,
                'random_minimize':None}
    
    
    def _fit(self, observables, labels, impurity_fn=TF.gini_impurity,
              **hyparams):
        self._hparams.update(hyparams)
        self._tree = _BDTF.grow_tree(observables, labels, impurity_fn,
                                     **self._hparams)
    
    def _predictions(self, observables, label=None):
        return np.fromiter( (_BDTF.predict_tree(datum, self.tree)
                             for datum in observables ),
                           float, len(observables))
    
    
    def show(self, spacing=' ', trimmed=True, rounded=True):
        _BDTF.show_tree(self.tree, 'Root', spacing, trimmed, rounded)
    
    def get_depth(self):
        return _BDTF.tree_depth(self.tree)
    
    def get_probabilities(self):
        probabilities = _BDTF.leaf_probabilities(self.tree)
        return sorted(probabilities, key=lambda e: (e is not None, e))
    
    def get_path(self, probability):
        root_node = _BDTF.get_probability_path(self.tree, probability)
        return _BDTC.TreePath(root_node, probability)
    
    def get_num_leaves(self):
        return _BDTF.get_num_leaves(self.tree)
    
    def contains_None(self):
        return _BDTF.tree_contain_None(self.tree)
    
    
    @property
    def tree(self): return self._tree
    
    # Define how classifier is indexed.
    def __getitem__(self, key):
        
        # If indexed, return indexed root node.
        return self.tree[key]
   

class _BinaryTreeClassifier(_BinaryTreeEstimator, _BC.BaseClassifier):
    __sclass__ = "BinaryTreeClassifier"