### Functions ###
def hist_normalise(histogram, bin_width):
    total_area = 0.0
    for hist_bin in histogram:
        total_area += hist_bin * bin_width
    return histogram / total_area


def check_label_params(*results, observables=False):
    if observables is False:
        labels, labels_pred = results[:2]
        train_results = results[2:]
    
        if len(labels) !=  len(labels_pred):
            raise ValueError(f"Your labels and labels_pred should be \
            the same length. You have given lengths {len(labels)} and \
            {len(labels_pred)}, respectively.")

        if (len(train_results) != 2) or (len(train_results) != 0):
            raise ValueError(f"Your must give both the test_labels and \
            test_labels_pred IN THIS ORDER, on neither. You have given \
            {len(train_results)} parameter(s).")

        if (len(train_results) == 2):
            if len(train_results[0]) != len(train_results[1]):
                raise ValueError(f"Your train_labels and train_labels_pred \
                should be the same length. You have given lengths \
                {len(train_results[0])} and {len(train_results[1])}, \
                respectively.")
    
    else:
        observables, labels, labels_pred = results[:3]
        train_results = results[3:]
        
        if (len(observables) != len(labels)) and (len(labels) !=  len(labels_pred)):
            raise ValueError(f"Your observables, labels and labels_pred should be \
            the same length. You have given lengths {len(observables)}, {len(labels)} \
            and {len(labels_pred)}, respectively.")

        if (len(train_results) != 3) or (len(train_results) != 0):
            raise ValueError(f"Your must give both the test_observables, test_labels \
            and test_labels_pred IN THIS ORDER, on neither. You have given \
            {len(train_results)} parameter(s).")

        if (len(train_results) == 3):
            if observables.shape[1] != train_results[0].shape[1]:
                raise ValueError(f"Your observables and train_observables, should \
                have the same legnth in the second dimension. You have given second \
                dimension lengths {observables.shape[1]} and \
                {train_results[0].shape[1]}, respectively.")
            
            
            if ((len(train_results[0]) != len(train_results[1]))
                and (len(train_results[1]) !=  len(train_results[0]))):
                
                raise ValueError(f"Your train_observables, train_labels and \
                train_labels_pred should be the same length. You have given lengths \
                {len(train_results[0])}, {len(train_results[1])} and \
                {len(train_results[2])}, respectively.")