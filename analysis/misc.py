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
            raise ValueError(f"Your labels and labels_pred should be " +
            f"the same length. You have given lengths {len(labels)} and " +
            f"{len(labels_pred)}, respectively.")

        if not ((len(train_results) == 2) or (len(train_results) == 0)):
            raise ValueError(f"Your must give both the test_labels and " +
            "test_labels_pred IN THIS ORDER, or neither. You have given " +
            f"{len(train_results)} parameter(s).")

        if (len(train_results) == 2):
            if len(train_results[0]) != len(train_results[1]):
                raise ValueError(f"Your train_labels and train_labels_pred " +
                "should be the same length. You have given lengths " +
                f"{len(train_results[0])} and {len(train_results[1])}, " +
                "respectively.")
    
    else:
        observables, labels, labels_pred = results[:3]
        train_results = results[3:]
        
        if (len(observables) != len(labels)) and (len(labels) !=  len(labels_pred)):
            raise ValueError(f"Your observables, labels and labels_pred should be " +
            f"the same length. You have given lengths {len(observables)}, {len(labels)} " +
            f"and {len(labels_pred)}, respectively.")

        if not ((len(train_results) == 3) or (len(train_results) == 0)):
            raise ValueError(f"Your must give both the test_observables, test_labels " +
            "and test_labels_pred IN THIS ORDER, or neither. You have given " +
            f"{len(train_results)} parameter(s).")

        if (len(train_results) == 3):
            if observables.shape[1] != train_results[0].shape[1]:
                raise ValueError(f"Your observables and train_observables, should " +
                "have the same legnth in the second dimension. You have given second " +
                f"dimension lengths {observables.shape[1]} and " +
                f"{train_results[0].shape[1]}, respectively.")
            
            
            if ((len(train_results[0]) != len(train_results[1]))
                and (len(train_results[1]) !=  len(train_results[0]))):
                
                raise ValueError(f"Your train_observables, train_labels and " +
                "train_labels_pred should be the same length. You have given lengths " +
                f"{len(train_results[0])}, {len(train_results[1])} and " +
                f"{len(train_results[2])}, respectively.")
                
                
def print_cm(confusion_matrix):
    """Print the confusion matrix to stdout."""
    tn, fp, fn, tp = confusion_matrix
    
    s_total = float(fn + tp)
    b_total = float(tn + fp)
    a, b, c, d = tn/b_total, fp/b_total, fn/s_total, tp/s_total

    print( "          -------------------------")
    print( "          |         Actual        |")
    print( "          |-----------------------|")
    print( "          |     S     |     B     |")
    print( "----------|-----------|-----------|")
    print( "|   |     |           |           |")
    print( "|   |     |  TP:      |  FP:      |")
    print(f"| P |  S  | {tp:7d}   | {fp:7d}   |")
    print(f"| r |     |   ({d:.2f})  |   ({b:.2f})  |")
    print( "| e |     |           |           |")
    print( "| d |-----|-----------|-----------|")
    print( "| i |     |           |           |")
    print( "| c |     |  FN:      |  TN:      |")
    print(f"| t |  B  | {fn:7d}   | {tn:7d}   |")
    print(f"|   |     |   ({c:.2f})  |   ({a:.2f})  |")
    print( "|   |     |           |           |")
    print( "-----------------------------------")