### Imports ###
import treefunctions as TF


### Classes ####
class TreeNode(object):
    
    # Initialise TreeNode.
    def __init__(self, labels, depth, impurity_fn):
        
        # Define attributes.
        self._depth = depth
        self._size = len(labels)
        
        # Find class sizes.
        self._classes = {'S': labels.sum(),  # Label is 0 or 1, for label
                         'B': len(labels) - labels.sum()}  #  in labels.
        
        # Calculate nodes impurity.
        self._impurity = impurity_fn(self._classes.values(), self._size)
    
    # Define how attributes are obtained.
    @property
    def depth(self): return self._depth
    @property
    def size(self): return self._size
    @property
    def classes(self): return self._classes
    @property
    def impurity(self): return self._impurity
        
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
                


class TreeLeaf(TreeNode):
                          
    # Initialise TreeLeaf.
    def __init__(self, P, observables, labels, depth, impurity_fn):
        
        # Define inherited attributes.
        super().__init__(labels, depth, impurity_fn)
        
        # Define singal probability attribute.
        self._P = P
        
    # Define how probability attribute is obtained.
    @property
    def P(self): return self._P
    
    # Define how leaf is shown as a string.
    def __str__(self, trimmed=True, rounded=True):
        
        # If rounded show values to 2 digits.
        digits = 2 if rounded else 16
        
        # Show singal probability and inherited properties, seperated by '|'.
        return f"P:{self.P:.{digits}f}| " + super().__str__(trimmed, rounded)

                    
        
class TreeBranch(TreeNode):
                          
    # Initialise TreeBranch.
    def __init__(self, split_value, observables, labels, depth, impurity_fn,
                 cost, n_obsv, **kwargs):
        
        # Define inherited attributes.
        super().__init__(labels, depth, impurity_fn)
        
        # Define branch attributes.
        self._split = split_value
        self._cost = cost
        self._n_obsv = n_obsv
        
        # Find partitions for this observable and partition value.
        partitions = TF.partition_data(observables, labels, split_value, n_obsv)
        
        # Find the subtrees from these partitions.
        self._subtrees = tuple(TF.grow_tree(observables, labels,
                                            impurity_fn = impurity_fn,
                                            depth = self._depth + 1,
                                            **kwargs)
                               for observables, labels in partitions)
        
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
        return (f"S:{self.split:.{digits}f}| O:{self.n_obsv:d}"
                + s_mid + "| " + super().__str__(trimmed, rounded))
    
    # Define how branch is indexed.
    def __getitem__(self, key):
        
        # If indexed, return indexed child node.
        return self.subtrees[key]