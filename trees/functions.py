### Imports ###
from scipy.optimize import minimize_scalar

import trees.nodes as TN



### Functions ###
def partition_data(observables, labels, partition_value, n_obsv):
    
    # Create condition for splitting data
    condition = observables[:,n_obsv] < partition_value
    
    # Create partitons from data.
    partitions = ((observables[condition], labels[condition]),   # Less than.
                  (observables[~condition], labels[~condition])) # Greater than.
    
    # Return partitions.
    return partitions


def gini_impurity(class_sizes, total_size):
    
    # Set total impurity to 1.
    G = 1.0
    
    # Iteratively minus each class' impurity from the total.
    for class_size in class_sizes:
        
        # Class' impurity is square of fraction of class in total .
        G -= (class_size / total_size)**2
    
    # Return total impurity.
    return G


def CART_cost(partitions, total_samples, impurity_fn=gini_impurity, **kwargs):
    
    # Find min samples size for a leaf.
    min_samples = kwargs.get('min_samples_leaf', None)
    # If None set to -1, for comparisons.
    min_samples = -1 if (min_samples is None) else min_samples
    
    # Iterate through each partition
    cost = 0.0
    for observables, labels in partitions:
        
        # Check each partition's size is large enough to split.
        partition_samples = len(labels)
        if len(labels) < min_samples:
            # Otherwise return the difference plus one,
            # so this can me minimizied towards an allowed value.
            return (min_samples - partition_samples + 1)
        
        # Find number of labels with value 1 and value 0 [when
        # each label is either 1 or 0].
        partition_values = (labels.sum(), partition_samples - labels.sum())
        
        # Find the impurity for the partition.
        partition_impurity = impurity_fn(partition_values, partition_samples)
        
        # Add to the running cost for these partitions
        cost += (partition_samples / total_samples) * partition_impurity
    
    # Return the cost for these partitions.
    return cost


def find_partition(observables, labels, size, impurity_fn, kwargs):
    """returns (n_observable, partition_value, CART_cost)"""
    
    # Create dictionary to hold optimal partition values for
    # each observable.
    optimize_results = {}
    
    # Find the range over each observable.
    v_mins = observables.min(axis=0)
    v_maxs = observables.max(axis=0)
    
    # Iterate over each observable to find be partition value.
    for n_obsv in range(observables.shape[1]):

        # Define the cost function to minimize by partition value
        def cost_fn(partition_value):
            
            # Partition data on partition value.
            partitions = partition_data(observables, labels,
                                        partition_value, n_obsv)
            
            # Return the CART cost of this partition.
            return CART_cost(partitions, size, impurity_fn=impurity_fn,
                             **kwargs)

        # Minimize the cost function to find optimal partition value
        optimize_result = minimize_scalar(cost_fn, method='Bounded',
                                          bounds=(v_mins[n_obsv],
                                                  v_maxs[n_obsv]))
        
        # Add (observable index, and optimal partition value) to dictionary,
        # indexed by their CART cost.
        optimize_results[optimize_result.fun] = (n_obsv, optimize_result.x)
    
    # Find the minimum CART cost from the optimal partitions.
    min_cost = min(optimize_results.keys())
    
    # Return the optimal partitions:
    return (optimize_results[min_cost][0], # observable index,
            optimize_results[min_cost][1], # partition value,
            min_cost)                      # and CART cost.


def grow_tree(observables, labels, impurity_fn=gini_impurity, depth=0, **kwargs):
    """observables = numpy.array( [[observable1_1, observable2_1, ...],
                                  [observable1_2, observable2_2, ...],
                                  ...] )
       labels = numpy.array( [label_1, label_2, ...] )
    
    kwargs:
        - max_depth: None (defualt) or int
        - min_samples_split: None (defualt) or int
        - min_samples_leaf: None (defualt) or int"""
    
    # Ensure observables and labels match-up.
    size = float(len(observables))
    if not len(observables) == len(labels):
        raise ValueError(f"Observables and labels must have the same first \
        dimension. You have given {len(observables)} and {len(labels)}.")
    
    # Calculate the number of 'True' labels, if none create a leaf.
    n_true = labels.sum() # Each label is either 0 or 1.
    if n_true == 0:
        return TN.TreeLeaf(0.0, observables, labels, depth, impurity_fn)
    
    
    # Check is passed max depth of the tree.
    if kwargs.get('max_depth') == depth:
        return TN.TreeLeaf(n_true / size, observables, labels, depth,
                        impurity_fn)
    # Check we have enough sample to partition.
    if kwargs.get('min_samples_split'):
        if size < kwargs.get('min_samples_split'):
            return TN.TreeLeaf(n_true / size, observables, labels, depth,
                            impurity_fn)
    
    
    # Calculate the number of 'False' labels, if none create a leaf.
    n_false = size - n_true
    if n_false == 0:
        return TN.TreeLeaf(1.0, observables, labels, depth, impurity_fn)
    
    # Find the best observable, its value, and its cost, to partition
    # samples on.
    n_obsv, partition_value, cost = find_partition(observables, labels,
                                                   size, impurity_fn,
                                                   kwargs)
    
    # If it can't split (e.g. min_leaf_sample to high), than the cost
    # is greater than 1, so create a leaf.
    if cost > 1.0:
        return TN.TreeLeaf(float(n_true / size), observables, labels, depth,
                        impurity_fn)
    
    # Else create a branch with this observable and partition value.
    return TN.TreeBranch(partition_value, observables, labels,
                         depth, impurity_fn, cost, n_obsv, **kwargs)


def show_tree(node, prefix='Root', spacing=' ', trimmed=True, rounded=True):
    
    # Print depth-spacing, node's type (prefix), and node's info.
    print((spacing * node.depth) + prefix
          + ':{' + node.__str__(trimmed, rounded) + '}')
    
    # If node is a branch, repeat for child nodes.
    if isinstance(node, TN.TreeBranch):
        show_tree(node[0], 'LB', spacing, trimmed, rounded)
        show_tree(node[1], 'GB', spacing, trimmed, rounded)

                
def tree_classify(datum, tree):
    
    # Check if node is a branch.
    if isinstance(tree, TN.TreeBranch):
        
        # If a branch, check if observable is
        # less than or equal to partition value.
        if datum[tree.n_obsv] <= tree.split:
            
            # If less, check datum against child node 0.
            return tree_classify(datum, tree[0])
        
        else:
            
            # If greater, check datum against child node 1.
            return tree_classify(datum, tree[1])
    else:
        
        # If not a branch, return signal probability.
        return tree.P

    
def tree_depth(tree):
    
    # Check if node is a branch.
    if isinstance(tree, TN.TreeBranch):
        
        # If a branch, find depths of branches.
        dl = tree_depth(tree[0])  # Less than branch.
        dg = tree_depth(tree[1])  # Greater than branch.
        
        # Find largest depth and increment by one.
        return (dl + 1) if (dl >= dg) else (dg + 1)
    
    else:
        
        # If a leaf, return zero.
        return 0