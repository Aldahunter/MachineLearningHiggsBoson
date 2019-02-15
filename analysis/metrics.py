import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import confusion_matrix, roc_auc_score

import analysis.misc as AMI

### Functions ###
def confusion_matrix(labels, labels_predicted, p_threshold=0.5):
    bool_predicted = np.fromiter((label_pred >= p_threshold
                                  for label_pred in labels_predicted),
                                dtype=bool, count=len(labels_predicted))
    return metrics.confusion_matrix(labels, bool_predicted).ravel()


def accuracy(tn, fp, fn, tp):
    return (tp + tn) / (tp + fp + fn + tn)


def precision(tn, fp, fn, tp):
    return tp / (tp + fp)


def recall(tn, fp, fn, tp):
    return tp / (tp + fn)


def fpr(tn, fp, fn, tp):
    """False Positive Rate"""
    return fp / (fp + tn)


def f1_score(tn, fp, fn, tp):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return (2 * p * r) / (p + r)


def auc(labels, labels_pred, **kwargs):
    return metrics.roc_auc_score(labels, labels_pred, **kwargs)


def find_p_threshold(labels, labels_pred):
    
    # Define function to minimize for p_threshold to give maximum Signal-Noise ratio.
    def minimise_fn(p_threshold):
        cm = confusion_matrix(labels, labels_pred, p_threshold)
        epsilon_b, epsilon_s = fpr(*cm), recall(*cm)
        
        try:
            score = -1.0 * (epsilon_s / np.sqrt(epsilon_b))
        except ZeroDivisionError:
            score = 0.0
        return score

    # Minimize the defined function.
    optimize_result = minimize_scalar(minimise_fn, method='Bounded', bounds=(0.0, 1.0))
    
    # Return optimized value of p_threshold.
    return optimize_result.x


def S_B_ratio(labels, labels_predicted, p_threshold=0.5):
    # Define physics values.
    sigma_s = 0.204436505
    sigma_b = 12.25108192
    luminosity = 100
    
    # Obtain Machine Learning values.
    cm = confusion_matrix(labels, labels_predicted, p_threshold)
    epsilon_b, epsilon_s = fpr(*cm), recall(*cm)
    
    # Calculate Signal-Noise ratio.
    try:
        sb_ratio = ( (sigma_s * np.sqrt(luminosity) * epsilon_s) 
                     / np.sqrt(sigma_b * epsilon_b) )
    except ZeroDivisionError:
        sb_ratio = 0.0
    
    # Return Signal-Noise ratio.
    return sb_ratio


def rms_deviation(labels, labels_pred, train_labels, train_labels_pred, n_bins=50):
    AMI.check_label_params(labels, labels_pred, train_labels, train_labels_pred)
    
    full_results = {'Test': (labels, labels_pred),
                    'Train': (train_labels, train_labels_pred)}
    
    histograms = {}
    for key, type_results in full_results.items():
        results = {'s': [], 'b': []}

        is_signal = type_results[0]
        signal_probabilities = type_results[1]
        results = {'s': signal_probabilities[is_signal is True],
                   'b': signal_probabilities[is_signal is False]}

        bins = np.linspace(0, 1, n_bins + 1)
        histogram = {'s': np.histogram(results['s'], bins)[0],
                     'b': np.histogram(results['b'], bins)[0]}
        
        width = bins[1] - bins[0]
        histogram['s'] = AMI.hist_normalise(histogram['s'], width)
        histogram['b'] = AMI.hist_normalise(histogram['b'], width)
        
        histograms[key] = histogram
        
    rmsq = {}
    for sb in ['b', 's']:
        deviation = (histograms['Train'][sb] - histograms['Test'][sb])**2
        rmsq[sb] = np.sqrt( deviation.sum() / len(deviation) )
    
    return rmsq


def get_scores(labels, labels_pred, *train_results, p_threshold=0.5, n_bins=50):
    AMI.check_label_params(labels, labels_pred, *train_results)
    
    scores = {}
    tn, fp, fn, tp = confusion_matrix(labels, labels_pred, p_threshold)

    scores["a"] = accuracy(tn, fp, fn, tp)
    scores["p"] = precision(tn, fp, fn, tp)
    scores["r"] = recall(tn, fp, fn, tp)
    scores["f1"] = f1_score(tn, fp, fn, tp)
    scores["auc"] = auc(labels, labels_pred)
    scores["s/b"] = S_B_ratio(labels, labels_predicted, p_threshold=0.5)
    
    if len(train_results) == 2:
        scores["rmsd"] = rms_deviation(labels, labels_pred, *train_results,
                                       n_bins=n_bins)

    return scores