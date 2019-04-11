"""A module to hold the classes associated with a Binary Decision Tree \
classifier."""

### Imports ###
from copy import deepcopy

import dwrangling.dataframes as DWDF
import trees._BinaryDecisionTree._BDTFuncs as _BDTF


### Classes ####
class NoneNode(object):
    def __init__(self, depth=None):
        # Define attributes.
        self._depth = depth
    
    def __repr__(self, *args, **kwargs):
        return "None"
    
    def __str__(self, *args, **kwargs):
        return "NoneType Node"
    
    def __bool__(self):
        return False
    
    def __eq__(self, other):
        return None == other
    
    def __neq__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash(None)
    
    @property
    def depth(self): return self._depth


    
class TreeNode(object):
    
    # Initialise TreeNode.
    def __init__(self, labels, depth, sample_weights, impurity_fn):
        
        # Define attributes.
        self._depth = depth
        self._size = len(labels)
        
        # Find class sizes.
        self._classes = {'S': labels.sum(),  # Label is 0 or 1, for label
                         'B': len(labels) - labels.sum()}  #  in labels.
        
        # Calculate nodes impurity.
        self._impurity = impurity_fn((sample_weights[labels].sum(),
                                      sample_weights[~labels].sum()),
                                      sample_weights.sum())
    
    # Define how attributes are obtained.
    @property
    def depth(self): return self._depth
    @property
    def size(self): return self._size
    @property
    def classes(self): return self._classes
    @property
    def impurity(self): return self._impurity
    
    def copy(self):
        return deepcopy(self)
        
    # Define how node is shown as a string.
    def __str__(self, trimmed=True, rounded=True):
        
        # If rounded show values to 2 digits.
        digits = 2 if rounded else 16
        
        # Define list of properties' string representations.
        propities = [f'I:{self.impurity:.{digits}f}',
                     f'D:{self.depth:d}',
                     f'T:{self.classes}',]
        
        # If trimmed show only the impurity.                  
        if trimmed:
            return propities[0]
        
        # Otherwise show all properties, seperated by '|'.
        return '| '.join(propities)

    def __repr__(self):
        return self.__str__()
                


class TreeLeaf(TreeNode):
                          
    # Initialise TreeLeaf.
    def __init__(self, P, observables, labels, depth, sample_weights,
                 impurity_fn):
        
        # Define inherited attributes.
        super().__init__(labels, depth, sample_weights, impurity_fn)
        
        # Define signal probability attribute.
        self._P = P
        
    # Define how probability attribute is obtained.
    @property
    def P(self): return self._P
    
    # Define how leaf is shown as a string.
    def __str__(self, trimmed=True, rounded=True):
        
        # If rounded show values to 2 digits.
        digits = 2 if rounded else 16
        
        # Show singal prob. and inherited properties, seperated by '|'.
        return f"P:{self.P:.{digits}f}| " + super().__str__(trimmed,
                                                            rounded)

    def __repr__(self):
        return self.__str__()

                    
        
class TreeBranch(TreeNode):
                          
    # Initialise TreeBranch.
    def __init__(self, split_value, observables, labels, depth,
                 sample_weights, impurity_fn, minimise_fn, cost,
                 n_obsv, **kwargs):
        
        # Define inherited attributes.
        super().__init__(labels, depth, sample_weights, impurity_fn)
        
        # Define branch attributes.
        self._split = split_value
        self._cost = cost
        self._n_obsv = n_obsv
        
        # Find partitions for this observable and partition value.
        partitions = _BDTF.partition_data(observables, labels,
                                          sample_weights,
                                          split_value, n_obsv)
        
        # Find the subtrees from these partitions.
        self._subtrees = tuple(_BDTF._grow_branch(*partition_set,
                                                  impurity_fn, minimise_fn,
                                                  self._depth + 1, **kwargs)
                               for partition_set in partitions)
        
    # Define how tree attributes are obtained.
    @property
    def split(self): return self._split
    @property
    def cost(self): return self._cost
    @property
    def n_obsv(self): return self._n_obsv
    @property
    def subtrees(self): return self._subtrees
    
    # Define how branch is shown as a string.
    def __str__(self, trimmed=True, rounded=True):
        
        # If rounded show values to 2 digits.
        digits = 2 if rounded else 16
        
        # If trimmed dont show cost.
        s_mid =  "" if trimmed else f"| C:{self.cost:.{digits}f}"
        
        # Show partition value (split), observable's index (n_obsv),
        # cost and inherited properties, seperated by '|'.
        return (f"S:{self.split:.{digits}f}| O:{self.n_obsv}"
                + s_mid + "| " + super().__str__(trimmed, rounded))
                     
    def __repr__(self):
        return self.__str__()
    
    def __getitem__(self, key):
        return self.subtrees[key]  #If indexed, index subtrees


class TreePath(object):
    
    def __init__(self, root_node, P):
        self._P = P
        self._tree = root_node
        self._leaves = _BDTF.get_num_leaves(root_node)
    
    def __str__(self):
        return f"TreePath(Probability: {self.P}; n_leaves: {self._leaves})"
    __repr__ = __str__

    def __getitem__(self, key):
        return self.tree[key]


    def show(self, spacing=' ', trimmed=True, rounded=True):
        """Prints the tree to stdout."""
        _BDTF.show_tree(self.tree, 'Root', spacing, trimmed, rounded)
    
    def get_depth(self):
        return _BDTF.tree_depth(self.tree)
    
    def get_paths(self):
        """Returns a :class:`tuple` of paths, where each path is a \
        :class:`tuple` of 1 character :class:`str` and each :class:`str` \
        is either '<' (less than branch) or '>' (greater than branch)."""  
        paths = _BDTF.get_split_order(self.tree, self.P)[-1]
        return [path[::-1] for path in paths]
    
    def get_observable_paths(self, observables):
        obs_index = {n_obs: obs for n_obs, obs in enumerate(observables)}
        obs_paths = _BDTF.get_observable_path(self.tree,
                                              self.P,
                                              obs_index)[-1]
        return [obs_path[::-1] for obs_path in obs_paths]
                     
    
    def get_samples(self, dataframe):
        """Returns a list of dataframes one for each leaf sample.
    
        Paramters:
         - dataframe: A :class:`dwrangling.dataframe.ODataFrame` to sample
                      from. 

        Returns:
         - samples: A :class:`list` of
                    :class:`dwrangling.dataframe.ODataFrame` dataframes, one
                    for each leaf node."""
                     
        if not isinstance(dataframe, DWDF.ODataFrame):
            raise ValueError("The 'dataframe' must be of type "
                             f"'{type(DWDF.ODataFrame)}', not "
                             f"'{type(dataframe)}'.")
        
        paths = self.get_paths()          
        return _BDTF.get_node_samples(self.tree, dataframe, paths)
    
    def get_path_sample_generators(self, dataframe):
        """Returns a list of generators one for each path to each leaf.
    
        Paramters:
         - dataframe: A :class:`dwrangling.dataframe.ODataFrame` to sample
                      from. 

        Returns:
         - samples: A :class:`tuple` :class:`generator`, where each
                    :class:`tuple` contains:
             + observable: A :class:`str` holding the observable which was
                           partitioned on;
             + partition_value: A :class:`float` for the value of the
                                observable used to partition the data;
             + branch: A 1 character :class:`str` to determine which branch
                       to take, '<' is the 'less than branch' and '>' is the
                       'greater than or equal to' branch;
             + sample: A :class:`dwrangling.dataframe.ODataFrame` holding
                       that node's sample."""
                     
        if not isinstance(dataframe, DWDF.ODataFrame):
            raise ValueError("The 'dataframe' must be of type "
                             f"'{type(DWDF.ODataFrame)}', not "
                             f"'{type(dataframe)}'.")
        
        generators = []
        for path in self.get_paths():
            generators.append(_BDTF.yield_path_samples(self.tree,
                                                       dataframe, path))
        return generators


    @property
    def P(self): return self._P
    @property
    def tree(self): return self._tree
    @property
    def leaves(self): return self._leaves