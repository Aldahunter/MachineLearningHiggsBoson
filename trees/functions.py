from copy import deepcopy
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
    """Grows (fits) a tree to the data given.
    
    Parameters:
     - observables = :class:`numpy.array` ( [[observable1_1, observable2_1, ...], \
     [observable1_2, observable2_2, ...], ...], dtype=:class:`float` );
     - labels = :class:`numpy.array` ( [label_1, label_2, ...], dtype=:class:`bool` );
     - kwargs:
        + max_depth: :class:`None` [Default] or :class:`int`;
        + min_samples_split: :class:`None` [Default] or :class:`int`;
        + min_samples_leaf: :class:`None` [Default] or :class:`int`.
    
    Returns:
     - Root: The root :class:`trees.nodes.TreeBranch` of the fitted tree."""
    
    # Ensure observables and labels match-up.
    size = float(len(labels))
    assert len(observables) == len(labels)
    if not len(observables) == len(labels):
        raise ValueError(f"Observables and labels must have the same first \
        dimension. You have given {len(observables)} and {len(labels)}.")
    
    # Calculate the number of 'True' labels, if none create a leaf.
    n_true = labels.sum() # Each label is either 0 or 1.
    if n_true == 0:
        return TN.TreeLeaf(0.0, observables, labels, depth, impurity_fn)
    
    if n_true / size == 1.0:
        print(f"n_sig: {n_true}, size :{len(labels)}, labs: {list(labels)}")
    
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
    n_false = (labels == 0).sum()
    if n_false == 0:
        print(f"n_sig: {n_true}, size :{len(labels)}, labs: {list(labels)}")
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
    
    # Else if node is leaf, return signal probability
    elif isinstance(tree, TN.TreeLeaf):
        return tree.P
    
    # Otherwise return 'NoneNode' object
    else:
        return TN.NoneNode(tree.depth)

    
    
def no_fn(node):
    """Return returns a :class:`trees.nodes.NoneNode`."""
    return TN.NoneNode(node.depth)

def recurse_tree(node, *args, branch_fn=no_fn, leaf_fn=no_fn, none_fn=no_fn):
    """Applys branch_fn to TreeBranch nodes, leaf_fn to TreeLeaf nodes and \
    none_fn to NoneNodes. To make recursive, ensure the branch_fn calls the \
    function it is defined within. [Default :func:`no_fn`]."""
    
    if isinstance(node, TN.TreeBranch):
        return branch_fn(node, *args)
    elif isinstance(node, TN.TreeLeaf):
        return leaf_fn(node, *args)
    elif node == None:
        return none_fn(node, *args)
    else:
        raise ValueError("Node must be of type TreeBranch, TreeLeaf " +
                         f"or NoneNode, not type '{type(node)}'.")


def show_tree(node, prefix='Root', spacing=' ', trimmed=True, rounded=True):
    """Prints the tree's structure to stdout."""
    
    print((spacing * node.depth) + prefix
          + ':{' + node.__str__(trimmed, rounded) + '}')
    
    def branch_fn(node):
        show_tree(node[0], 'LB', spacing, trimmed, rounded)
        show_tree(node[1], 'GB', spacing, trimmed, rounded)
        return
    
    return recurse_tree(node, branch_fn=branch_fn)


def tree_depth(node):
    """Returns the depth (height) or the tree."""
    
    def branch_fn(node):
        # If a branch, find height of branches.
        hl = tree_depth(node[0])  # Less than branch.
        hg = tree_depth(node[1])  # Greater than branch.
        
        # If both are None return None
        if (hl == None) and (hg == None):
            return TN.NoneNode(node.depth)
        
        # If one branch is None, return the other incremented by one.
        if hl == None:
            return hg + 1
        elif hg == None:
            return hl + 1
        
        # Otherwise return largest height incremented by one.
        else:
            return (hl + 1) if (hl >= hg) else (hg + 1)
    
    def leaf_fn(node):
        return 0
    
    return recurse_tree(node, branch_fn=branch_fn, leaf_fn=leaf_fn)

    
def leaf_probabilities(node):
    """Returns a set of all possible probabilites that can \
    come from the tree."""
    
    def branch_fn(node):
        # If a branch, find probabilities sets of each.
        pl = leaf_probabilities(node[0])  # Less than branch.
        pg = leaf_probabilities(node[1])  # Greater than branch.
        
        # Find combine and sort the sets.
        return set(sorted(pl | pg, key=lambda e: (e is not None, e)))
    
    def leaf_fn(node):
        return {node.P,}
    
    return recurse_tree(node, branch_fn=branch_fn, leaf_fn=leaf_fn)


def get_probability_path(node, probability):
    """Returns a tree with only the nodes with trace the \
    path(s) to the given probability leaf(s)."""
    
    def branch_fn(node):
        # If a branch, find probabilities sets of each.
        tl = get_probability_path(node[0], probability)  # Less than branch.
        tg = get_probability_path(node[1], probability)  # Greater than branch.
        
        # If no leafs with the probability return NoneNode
        if (tl == None) and (tg == None):
            return TN.NoneNode(node.depth)
        
        # Create list to hold subtrees
        subtrees = []
        subtrees.append(tl)
        subtrees.append(tg)
        
        # Coppy current tree and update correct path subtrees
        node = deepcopy(node)
        node._subtrees = tuple(subtrees)
        
        # Return the current tree
        return node
    
    def leaf_fn(node):
        # If probability is correct
        if node.P == probability:
            
            # Return a copy of the leaf
            return deepcopy(node)
        
        # Otherwise return None
        return TN.NoneNode(node.depth)
    
    return recurse_tree(node, branch_fn=branch_fn, leaf_fn=leaf_fn)

    
def tree_contain_None(node):
    """Checks if a tree contains 'None' for a node."""
    
    def branch_fn(node):
        # If a branch check subtrees
        nl = tree_contain_None(node[0])  # Less than branch.
        ng = tree_contain_None(node[1])  # Greater than branch.
        
        # If one branch is True return True
        return (nl or ng)
    
    def none_fn(node):
        return True
    
    def leaf_fn(node):
        return False
    
    return recurse_tree(node, branch_fn=branch_fn, leaf_fn=leaf_fn,
                        none_fn=none_fn)


def get_num_leaves(node):
    """Finds the number of leaves in a tree, excludes NoneNodes."""
    
    def branch_fn(node):
        ll = get_num_leaves(node[0])
        lg = get_num_leaves(node[1])
        
        return ll + lg
    
    def leaf_fn(node):
        return 1
    
    def none_fn(node):
        return 0
    
    return recurse_tree(node, branch_fn=branch_fn, leaf_fn=leaf_fn,
                        none_fn=none_fn)


def get_split_order(node, P):
    
    def branch_fn(node):
        sl = get_split_order(node[0], P)
        sg = get_split_order(node[1], P)
        
        if (not sl[0]) and (not sg[0]):
            return (False, None)
        
        paths = []
        if sl[0]:
            for n_path, _ in enumerate(sl[1]):
                sl[1][n_path] += ('<',)
            paths += sl[1]
        if sg[0]:
            for n_path, _ in enumerate(sg[1]):
                sg[1][n_path] += ('>',)
            paths += sg[1]
        
        return (True, paths)
    
    def leaf_fn(node):
        if node.P == P:
            return (True, [()])
        else:
            return (False, None)
            
    def none_fn(node):
        return (False, None)
    
    return recurse_tree(node, branch_fn=branch_fn, leaf_fn=leaf_fn,
                        none_fn=none_fn)


def get_node_samples(node, dataframe, paths):
    """Traces a path to obtain a leaf's sample from the dataframe.
    
    Paramters:
     - node: A root :class:`trees.nodes.TreeBranch` to climb to the leaf \
     nodes. (To ignore nodes, replace them with \
     :class:`trees.nodes.NoneNonde`);
     - dataframe: A :class:`dwrangling.dataframe.ODataFrame` to sample from;
     - paths: A 2D :class:`tuple` of 1 character :class:`str`, where each \
     string must be either '<' (for the less than branch) or '>' (for the \
     greater than branch). (This can be obtained from :func:`get_split_order`). 
    
    Returns:
     - samples: A :class:`list` of :class:`dwrangling.dataframe.ODataFrame` \
     dataframes, one for each leaf node."""
    
    # Copy to create root values
    root_node = node.copy()
    root_dataframe = dataframe.copy()
    
    # Obtain observable names from dataframe
    observables = root_dataframe.columns
    
    # Iterate over paths to obtain node sample sizes
    samples = []
    for path in paths:
        
        # Reset node and dataframe for each path
        node = root_node
        dataframe = root_dataframe
        
        # Iterate over every branch decision in the path
        for branch in path:
            
            # If node isn't a branch path is too long as can't split
            if not isinstance(node, TN.TreeBranch):
                raise ValueError("Path is too long for tree.")
            # If incorrect branch symbol given raise error
            if not branch in ['<', '>']:
                raise ValueError("Path elements must be '>' or '<', " +
                                 f"not '{branch}'.")
            
            # If branch is '<' take the 'less than' samples and child 
            # node, otherwise take 'greater than or equal to'.
            br = 0 if (branch is '<') else 1

            # Sample dataframe and climb to child node
            dataframe = dataframe.partition(observables[node.n_obsv],
                                            node.split)[br]
            node = node[br]
        
        # Add final sampled dataframe of the path to samples
        samples.append(dataframe)
        
    # Return all path sampled dataframes
    return samples


def yield_path_samples(node, dataframe, path):
    """Returns a generator of the sampled dataframe for each node in the path.
    
    Paramters:
     - node: A root :class:`trees.nodes.TreeBranch` to climb to the leaf \
     nodes. (To ignore nodes, replace them with \
     :class:`trees.nodes.NoneNonde`);
     - dataframe: A :class:`dwrangling.dataframe.ODataFrame` to sample from;
     - path: A :class:`tuple` of 1 character :class:`str`, where each \
     string must be either '<' (for the less than branch) or '>' (for the \
     greater than branch). (This can be obtained from :func:`get_split_order`). 
    
    Returns:
     - samples: A :class:`tuple` :class:`generator`, where each :class:`tuple` contains:
         + observable: A :class:`str` holding the observable which was partitioned on;
         + partition_value: A :class:`float` for the value of the observable used to\
         partition the data;
         + branch: A 1 character :class:`str` to determine which branch to take, '<' is \
         the 'less than branch' and '>' is the 'greater than or equal to' branch;
         + sample: A :class:`dwrangling.dataframe.ODataFrame` holding that node's sample."""
    
    # Obtain observable names from dataframe
    observables = dataframe.columns
    
    # Iterate over every branch decision in the path
    for branch in path:
        
        # If node isn't a branch path is too long as can't split
        if not isinstance(node, TN.TreeBranch):
            raise ValueError("Path is too long for tree.")
        # If incorrect branch symbol given raise error
        if not branch in ['<', '>']:
            raise ValueError("Path elements must be '>' or '<', " +
                             f"not '{branch}'.")
        
        # Yield initial dataframe and where it is split.
        yield (observables[node.n_obsv], branch, node.split, dataframe)
        
        # If branch is '<' take the 'less than' samples and child 
        # node, otherwise take 'greater than or equal to'.
        br = 0 if (branch is '<') else 1

        # Sample dataframe and climb to child node
        dataframe = dataframe.partition(observables[node.n_obsv],
                                        node.split)[br]
        node = node[br]