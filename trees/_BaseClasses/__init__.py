"""A library containing the base classifier."""
import numpy as np

import trees._BaseClasses._BaseClassesFuncs as _BCF
import trees.functions as TF


class BaseEstimator(object):
    __sclass__ = "BaseEstimator"
    _hparams = {}
    _fitted = False
    
    
    def __init__(self, **hyparams):
        self._hparams.update(hyparams)
    
    def set_params(self, **hparams):
        self._hparams.update(hparams)
    set_hyparams = set_params
    
    def get_params(self, deep=True):
        return self._hparams
    get_hyparams = get_params
    
    
    def fit(self, observables, labels, *args, sample_weights=None,
            impurity_fn=TF.gini_impurity, normalise_weights=True, **kwargs):
        
        observables, labels, sample_weights = \
                     _BCF.sanity_checks(observables, labels, sample_weights,
                                        normalise_weights)
        self._fit(observables, labels, sample_weights, impurity_fn,
                  *args, **kwargs)
        self._fitted = True
    
    def predict(self, observables, labels=None, *args, **kwargs):
        
        observables = np.asfortranarray(observables)
        predictions = self._predictions(observables, *args, **kwargs)
        return predictions
    predict_prob = predict
    
    
    def _fit(self, observables, labels, sample_weights, impurity_fn,
             *args, **kwargs):
        raise NotImplementedError("Method '_fit'.")
    
    def _predictions(self, observables, *args, **kwargs):
        raise NotImplementedError("Method '_predictions'.")
    
    
    def __str__(self):
        return (f"{self.__sclass__}(Fitted: {self._fitted}; "
                + self._hparams.__repr__()[1:-1] + ")")
    def __repr__(self):
        return '<' + self.__str__() + '>'
    
    def __getitem__(self, key):
        return self._hparams[key]
    
    

class BaseClassifier(BaseEstimator):
    __sclass__ = "BaseClassifier"
    _theshold = None
    
    def __init__(self, **hyparams):
        self._hparams['train_val_frac'] =  0.75
        BaseEstimator.__init__(self, **hyparams)
    
    
    def fit(self, observables, labels, *args, sample_weights=None,
            impurity_fn=TF.gini_impurity, normalise_weights=True, **kwargs):
        
        observables, labels, sample_weights = \
                     _BCF.sanity_checks(observables, labels, sample_weights,
                                        normalise_weights)
        t, v = self._split_data(labels, self._hparams['train_val_frac'])
        self._fit(observables[t], labels[t], sample_weights[t], impurity_fn,
                  *args, **kwargs)
        predictions = self.predict_prob(observables[v], *args, **kwargs)
        self._theshold = self._get_threshold(labels[v], predictions)
        self._fitted = True

    def predict_prob(self, observables, labels=None, *args, **kwargs):
        observables = np.asfortranarray(observables)
        probabilities = self._predictions(observables, *args, **kwargs)
        assert isinstance(probabilities, np.ndarray)
        return probabilities
    
    def predict_from_prob(self, probabilities, *args, **kwargs):
        assert isinstance(probabilities, np.ndarray)
        return (probabilities > self.threshold).astype(float)
    
    def predict(self, observables, labels=None, *args, **kwargs):
        probabilities = self.predict_prob(observables, *args, **kwargs)
        return self.predict_from_prob(probabilities)
    
    
    @staticmethod
    def _split_data(labels, train_val_frac):
        return _BCF._test_validate_split(labels, train_val_frac)
    @staticmethod
    def _get_threshold(labels, predictions):
        return _BCF._threshold(labels, predictions)
          
        
    @property
    def threshold(self): return self._theshold
    