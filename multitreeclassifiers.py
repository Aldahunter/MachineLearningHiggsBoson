### Imports ###
import random
from collections import Counter
from scipy.optimize import minimize, minimize_scalar
from analysis import accuracy, precision, recall, f1_score, confusion_matrix


### Functions ###
def df_to_ML_input(df):
    ML_input = []
    
    for index, row in df.loc[:, df.columns != 'signal'].iterrows():
        ML_input.append( (tuple(row), df.signal[index]) )
    
    return tuple(ML_input)

def partition_data(data, partition_value, n_obsv):
    partitions = ([], [])
    for datum in data:
        n = 0 if (datum[0][n_obsv] <= partition_value) else 1
        partitions[n].append(datum)
    return partitions

def gini_impurity(partition_samples, total_samples):
    G = 1.0
    for partition_sample in partition_samples:
        G -= (partition_sample / total_samples)**2
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
    """returns (n_observable, partition_value, CART_cost)"""
    optimize_results = {}
    
    for n_obsv in range(len(data[0][0])):
        
        v_min = min(data, key=lambda datum: datum[0][n_obsv])[0][n_obsv]
        v_max = max(data, key=lambda datum: datum[0][n_obsv])[0][n_obsv]
#         init_guess = random.uniform(v_min, v_max)

        def cost_fn(partition_value):
            partitions = partition_data(data, partition_value, n_obsv)
            return CART_cost(partitions, size, impurity_fn=impurity_fn, **kwargs)

#         optimize_result = minimize(cost_fn, init_guess, method='Nelder-Mead')
        optimize_result = minimize_scalar(cost_fn, method='Bounded', bounds=(v_min, v_max))

#         optimize_results[optimize_result.fun] = (n_obsv, optimize_result.x[0])
        optimize_results[optimize_result.fun] = (n_obsv, optimize_result.x)
    
    min_cost = min(optimize_results.keys())
    return optimize_results[min_cost][0], optimize_results[min_cost][1], min_cost


def grow_tree(data, impurity_fn=gini_impurity, depth=0, **kwargs):
    """data = [(values, label), ...]
       values = (observable1, observable2, ...)
    
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
    
    n_obsv, partition_value, cost = find_partition(data, size,
                                                   impurity_fn,
                                                   kwargs)
    
    # Can't split (e.g. min_leaf_sample to high)
    if cost > 1.0:
        return TreeLeaf(float(n_true / size), data, depth, impurity_fn)
    
    return TreeBranch(partition_value, data, depth,
                      impurity_fn, cost, n_obsv, **kwargs)


def show_tree(node, prefix='Root', spacing=' ', trimmed=True, rounded=True):
    print(spacing * node.depth + prefix
          + ':{' + node.__str__(trimmed, rounded) + '}')
    if isinstance(node, TreeBranch):
        show_tree(node[0], 'LB', spacing, trimmed, rounded)
        show_tree(node[1], 'GB', spacing, trimmed, rounded)

                
def tree_classify(datum, tree):
    """datum = (values, label)"""
    if isinstance(tree, TreeBranch):
        if datum[0][tree.n_obsv] <= tree.split:
            return tree_classify(datum, tree[0])
        else:
            return tree_classify(datum, tree[1])
    else:
        return tree.P

def tree_depth(tree):
    """Find depth"""
    if isinstance(tree, TreeBranch):
        dl = tree_depth(tree[0])
        dg = tree_depth(tree[1])
        if dl >= dg:
            return dl + 1
        else:
            return dg + 1
    else:
        return 0



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
    def __init__(self, split, samples, depth, impurity_fn, cost, n_obsv,
                 **kwargs):
        super().__init__(samples, depth, impurity_fn)
        
        self._split = split
        self._cost = cost
        self._n_obsv = n_obsv
        
        partitions = partition_data(samples, split, n_obsv)
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
    def n_obsv(self): return self._n_obsv
    @property
    def subtrees(self): return self._subtrees
    
    def __str__(self, trimmed=True, rounded=True):
        digits = 2 if rounded else 16
        s_mid =  "" if trimmed else f"| C:{self.cost:.{digits}f}"
        
        return (f"S:{self.split:.{digits}f}| O:{self.n_obsv:d}"
                + s_mid + "| " + super().__str__(trimmed, rounded))
    
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
    train = plant_tree
    
    def show_tree(self, spacing=' ', trimmed=True, rounded=True):
        show_tree(self.tree, 'Root', spacing, trimmed, rounded)
    
    def classify(self, datum):
        return tree_classify(datum, self.tree)
    
    def get_depth(self):
        return tree_depth(self.tree)
    
    @property
    def tree(self): return self._tree