### Functions ###
def gini_impurity(class_weights, total_weight):
    
    # Set sum_square_weight to 0.
    sum_square_weight = 0.0
    
    # Iteratively minus each class' impurity from the total.
    for class_weight in class_weights:
        
        # Class' impurity is square of fraction of class in total .
        sum_square_weight += class_weight**2
    
    # Return total impurity.
    return 1.0 - sum_square_weight / total_weight**2

def entropy_impurity(class_weights, total_weight):
    pass