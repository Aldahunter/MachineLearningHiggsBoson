### Functions ###
def gini_impurity(class_sizes, total_size):
    
    # Set total impurity to 1.
    G = 1.0
    
    # Iteratively minus each class' impurity from the total.
    for class_size in class_sizes:
        
        # Class' impurity is square of fraction of class in total .
        G -= (class_size / total_size)**2
    
    # Return total impurity.
    return G

def entropy_impurity(class_sizes, total_size):
    pass