### Imports ###
import random
from collections import Counter
from scipy.optimize import minimize_scalar
from analysis import accuracy, precision, recall, f1_score, confusion_matrix


### Functions ###
def partition_data(data, partition_value):
    partitions = ([], [])
    for datum in data:
        n = 0 if (datum[0] <= partition_value) else 1
        partitions[n].append(datum)
    return partitions

def gini_impurity(partition_values, total_samples):
    G = 1.0
    for sample_value in partition_values:
        G -= (sample_value / total_samples)**2
    return G

def CART_cost(partitions, total_samples, impurity_fn=gini_impurity, **kwargs):
    if not kwargs.get('min_samples_leaf'):
        min_samples = -1
    else: 
        min_samples = kwargs.get('min_samples_leaf')
    
    cost = 0.0
    for partition in partitions:
        partition_samples = len(partition)
        if partition_samples < min_samples:
            return (min_samples - partition_samples + 1)
        
        partition_values = Counter([label for _, label in partition]).values()
        partition_impurity = impurity_fn(partition_values, partition_samples)
        
        cost += (partition_samples / total_samples) * partition_impurity
    return cost


def find_partition(data, size, impurity_fn, kwargs):
    v_min = min(data, key=lambda datum: datum[0])[0]
    v_max = max(data, key=lambda datum: datum[0])[0]
#     init_guess = random.uniform(v_min, v_max)
    
    def cost_fn(partition_value):
        partitions = partition_data(data, partition_value)
        return CART_cost(partitions, size, impurity_fn=impurity_fn, **kwargs)
    
#     optimize_result = minimize(cost_fn, init_guess, method='Nelder-Mead')
    print("New Search")
    optimize_result = minimize_scalar(cost_fn, method='Bounded', bounds=(v_min, v_max))
    
#     return optimize_result.x[0], optimize_result.fun
    return optimize_result.x, optimize_result.fun


def grow_tree(data, impurity_fn=gini_impurity, depth=0, **kwargs):
    """data = [(value, label), ...]
    
    kwargs:
        - max_depth: None or int
        - min_samples_split: None or int
        - min_samples_leaf: None or int"""
    
    size = len(data)    
    n_true = len([label for _, label in data if label])
    
    if kwargs.get('max_depth') == depth:
        return TreeLeaf(float(n_true / size), data, depth, impurity_fn)
    if kwargs.get('min_samples_split'):
        if size < kwargs.get('min_samples_split'):
            return TreeLeaf(float(n_true / size), data, depth, impurity_fn)
    
    
    n_false = size - n_true
    if n_true == 0: return TreeLeaf(0.0, data, depth, impurity_fn)
    if n_false == 0: return TreeLeaf(1.0, data, depth, impurity_fn)
    
    partition_value, cost = find_partition(data, size, impurity_fn, kwargs)
    
    # Can't split (e.g. min_leaf_sample to high)
    if cost > 1.0:
        return TreeLeaf(float(n_true / size), data, depth, impurity_fn)
    
    return TreeBranch(partition_value, data, depth,
                      impurity_fn, cost, **kwargs)


def show_tree(node, prefix='Root', spacing=' ', trimmed=True, rounded=True):
    print(spacing * node.depth + prefix
          + ':{' + node.__str__(trimmed, rounded) + '}')
    if isinstance(node, TreeBranch):
        show_tree(node[0], 'LB', spacing, trimmed, rounded)
        show_tree(node[1], 'GB', spacing, trimmed, rounded)

                
def tree_classify(datum, tree):
    """datum = (value, label)"""
    if isinstance(tree, TreeBranch):
        if datum[0] <= tree.split:
            return tree_classify(datum, tree[0])
        else:
            return tree_classify(datum, tree[1])
    else:
        return tree.P



### Classes ####
class TreeNode(object):
    def __init__(self, samples, depth, impurity_fn):
        self._depth = depth
        self._size = len(samples)
        self._classes = dict(Counter([label for _, label in samples]))
        self._impurity = impurity_fn(self._classes.values(), self._size)
    
    @property
    def depth(self): return self._depth
    @property
    def size(self): return self._size
    @property
    def classes(self): return self._classes
    @property
    def impurity(self): return self._impurity
        
    def __str__(self, trimmed=True, rounded=True):
        digits = 2 if rounded else 16
        
        propities = [f'I:{self.impurity:.{digits}f}',
                     f'D:{self.depth:d}',
                     #f'N:{self.size:d}',
                     f'L:{self.classes}',]
                          
        if trimmed:
            return propities[0]
        return '| '.join(propities)
                


class TreeLeaf(TreeNode):
    def __init__(self, P, samples, depth, impurity_fn):
        super().__init__(samples, depth, impurity_fn)
        
        self._P = P
        
    @property
    def P(self): return self._P
    
    def __str__(self, trimmed=True, rounded=True):
        digits = 2 if rounded else 16
        
        return f"P:{self.P:.{digits}f}| " + super().__str__(trimmed, rounded)

        
class TreeBranch(TreeNode):
    def __init__(self, split, samples, depth, impurity_fn, cost, **kwargs):
        super().__init__(samples, depth, impurity_fn)
        
        self._split = split
        self._cost = cost
        
        partitions = partition_data(samples, self._split)
        self._subtrees = tuple(grow_tree(partition,
                                          impurity_fn = impurity_fn,
                                          depth = self._depth + 1,
                                          **kwargs)
                               for partition in partitions)
        
    @property
    def split(self): return self._split
    @property
    def cost(self): return self._cost
    @property
    def subtrees(self): return self._subtrees
    
    def __str__(self, trimmed=True, rounded=True):
        digits = 2 if rounded else 16
        s_mid =  "" if trimmed else f"| C:{self.cost:.{digits}f}"
        
        return (f"S:{self.split:.{digits}f}" + s_mid + "| "
                + super().__str__(trimmed, rounded))
    
    def __getitem__(self, key):
        return self.subtrees[key]


class BinaryTreeClassifier(object):
    
    def plant_tree(self, train_data, impurity_fn=gini_impurity, **kwargs):
        """kwargs:
            - max_depth=None,
            - min_samples_split=None,
            - min_samples_leaf=None,
            - min_weight_fraction_leaf=None,
            - max_leaf_nodes=None,
            - max_features=None"""
        self._tree = grow_tree(train_data, impurity_fn, **kwargs)
    
    def show_tree(self, spacing=' ', trimmed=True, rounded=True):
        show_tree(self.tree, 'Root', spacing, trimmed, rounded)
    
    def classify(self, datum):
        return tree_classify(datum, self.tree)
    
    @property
    def tree(self): return self._tree