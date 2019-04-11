import numpy as np

import trees._BaseClasses as _BC
from trees._BinaryDecisionTree import _BinaryTreeClassifier as BinaryTreeC, _BinaryTreeEstimator as BinaryTreeE

# Small offsets for logarithimic equations
_LOGMIN = np.finfo(np.float64).eps
_LOGMAX = 1.0 - np.finfo(np.float64).epsneg

class _AdaBoostEstimator(_BC.BaseEstimator):
    """hyparams: - n_boosts = 100,
                 - learning_rate = 1.0,
                 - voting = 'soft', 
                 - weight_dist = 'weighted_impurity', ['bootstrap']
                 - algorithm = 'SAMME.R', ['SAMME']
                 - max_depth = 1,
                 - min_samples_leaf = None,
                 - min_samples_split = None."""
    __sclass__ = "AdaBoostEstimator"
    _hparams = {'n_boosts':100, 'learning_rate':1, 'voting':'soft',
                'weight_dist':'weighted_impurity', 'algorithm':'SAMME.R',
                'max_depth':1, 'min_samples_leaf':None,
                'min_samples_split':None}
    
    def _fit(self, observables, labels, sample_weights, impurity_fn,
             inform=False, fit_kwargs=None, predict_kwargs=None, **hyparams):
        self._hparams.update(hyparams)

        if predict_kwargs is None: predict_kwargs = {}
        if fit_kwargs is None: fit_kwargs = {}

        correct_weights = None
        if (self._hparams['voting'].lower() == 'soft'
        and self._hparams['algorithm'].lower() == 'SAMME'):
            correct_weights = lambda predictions: predictions > 0.5
        elif isinstance(self._hparams['voting'], float):
            if not 0.0 < self._hparams['voting'] < 1.0:
                raise ValueError("If 'voting' is a float, it must be a "
                                 "value between 0 and 1.")
            correct_weights \
                = lambda predictions: predictions > self._hparams['voting']
            self._hparams['voting'] = 'soft'     
        
        # If no sample weights equally weight each event
        if sample_weights is None:
            self.sample_weights = np.full(labels.size, 1.0/labels.size,
                                          dtype=np.float64)
        # Otherwise check all weights are positive, then normalise
        else:
            if np.any(sample_weights < 0):
                raise ValueError("You must have positive sample weights "
                                 "for the algorithm to work.")
            self.sample_weights = np.divide(sample_weights,
                                            sample_weights.sum(),
                                            dtype=np.float64)
        
        # Setup lists to hold classifiers and thier weights and errors
        n_boosts = self._hparams['n_boosts']
        self._classifiers = np.empty(n_boosts, dtype=object)
        self._vweight = np.full(n_boosts, 0, dtype=np.float64)
        
        # Retrive Classifier and algorithm from given hyperparameters
        Classifier = \
                 self._possible_classifiers[self._hparams['voting'].lower()]
        update_weights = \
                self._fitting_algorithms[self._hparams['algorithm'].lower()]
        
        # Iterate over each boost
        for n_boost in range(n_boosts):
        
            # Create instance of classifier
            classifier = Classifier(**self._hparams)
            
            # Fit classifier to observables by event weights
            classifier.fit(observables, labels, impurity_fn=impurity_fn,
                           sample_weights=self.sample_weights,
                           normalise_weights=False, **fit_kwargs)

            # Obtain the classifier predictions
            labels_predicted = classifier.predict(observables,
                                                  **predict_kwargs)
            if correct_weights:
                labels_predicted = correct_weights(labels_predicted)
            
            # Find the incorrecly classified events
            self.incorrect = labels != labels_predicted

            # Calculate the weigthed error of the classifier
            self.werror = self.sample_weights[self.incorrect].sum()
            ogerror = self.werror.copy()

            # If classifier is perfect stop training
            if self.werror <= 0:
                print(f"Perfect Classifier on Boost: {n_boost}")
                self._classifiers[n_boost] = classifier
                self._vweight[n_boost] = 1.0
                break
            if self.werror == 0.5:
                print("Classifier is as good as random guessing on Boost "
                      f"'{n_boost}' so cannot converge any more.")
                break

            # Update sample weights (algorithm dependent)
            voting_weight = update_weights(self, labels, labels_predicted)
            
            # Normalise the sample weights
            self.sample_weights /= self.sample_weights.sum()
            np.clip(self.sample_weights, _LOGMIN, None,
                    out=self.sample_weights)
            
            # Update classifier's weight
            self._classifiers[n_boost] = classifier
            self._vweight[n_boost] = voting_weight
            
            # Inform progress
            if inform:
                print(f"Boost: {n_boost+1} of {n_boosts}")
        
        # If finished early remove empty classifiers and weights
        if n_boost != n_boosts -1:
            boosts = self._classifiers != None
            self._classifiers = self._classifiers[boosts]
            self._vweight = self._vweight[boosts]
    
    
    def _fit_SAMME(self, labels, labels_predicted):   

        # Trim lower bound of predicted to smallest possible value
        # so that the log is defined.
        self.werror = np.clip(self.werror, _LOGMIN, _LOGMAX)

        # Calculate the classifier's weight for voting
        voting_weight = (self._hparams['learning_rate']
                         * np.log((1 - self.werror) / self.werror))
                
        # Update weights so correct predictions' weights are decreased
        # and incorrect weights are increased respective to the
        # classifiers voting weight
        self.sample_weights *= np.exp(np.where(self.incorrect, 1, -1)
                                      * voting_weight)
        
        # Return the calculated voting_weight
        return voting_weight
        
    def _fit_SAMMER(self, labels, labels_predicted):
        # Swap labels to  0 -> -1  &  1 -> +1  required for equations
        labels_coding = np.where(labels, 1, -1)

        # Trim lower bound of predicted to smallest possible value
        # so that the log is defined.
        np.clip(labels_predicted, _LOGMIN, _LOGMAX,
                out=labels_predicted)

        # Update sample weights
        self.sample_weights *= np.exp(-0.5 * labels_coding
                                      * self._hparams['learning_rate']
                                      * (np.log(labels_predicted)
                                         - np.log(1-labels_predicted)))

        # Set voting weight
        return 1.0
    
    # Dictionary for certain hyperparameter values
    _possible_classifiers = {'soft':BinaryTreeE, 'hard':BinaryTreeC}
    _fitting_algorithms = {'samme':_fit_SAMME, 'samme.r':_fit_SAMMER}
    
    
    def _predictions(self, observables, label=None, *args, **kwargs):

        # Initiate list to hold each classifier's predictions
        get_predictions = \
            self._predictions_algorithms[self._hparams['algorithm'].lower()]
        predictions = get_predictions(self, observables)

        # Get weighted average for prediction
        predictions /= self._vweight.sum()

        # Normalise logs to probabilities and return
        return 1 / (np.exp(-2 * predictions) + 1)
    
    def _predict_SAMME(self, observables):
        # Iterate over each classifier
        predictions = 0.0
        for weight, classifier in zip(self._vweight, self._classifiers):
            
            # Obtain predicitions
            labels_predicted = classifier.predict_prob(observables)

            # Sum the logged predictions of the classes
            if weight > 0:
                predictions +=  weight * labels_predicted
            else:
                predictions +=  -weight * (1-labels_predicted)
        
        # Return predictions
        return predictions
    
    def _predict_SAMMER(self, observables):
        # Iterate over each classifier
        predictions = 0.0
        for classifier in self._classifiers:

            # Obtain predicitions
            labels_predicted = classifier.predict_prob(observables)

            # Trim lower bound of predicted to smallest possible value
            # so that the log is defined.
            np.clip(labels_predicted, _LOGMIN, _LOGMAX,
                    out=labels_predicted)

            # Calculate the logged prediction of the classes  
            log_predicted = np.log(labels_predicted)
            log_predicted -= 0.5 * (log_predicted
                                    + np.log(1 - labels_predicted))

            # Sum the logged predictions of the classes
            predictions += log_predicted
        
        # Return predictions
        return predictions
    
    _predictions_algorithms = {'samme':_predict_SAMME,
                               'samme.r':_predict_SAMMER}
    
    
                      
    @property
    def classifiers(self): return self._classifiers
    @property
    def weights(self): return self._vweight
                               
        
    def __iter__(self):
        return iter(self._classifiers)



class _AdaBoostClassifier(_AdaBoostEstimator, _BC.BaseClassifier):
    __sclass__ = "AdaBoostClassifier"