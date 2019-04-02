### Imports ###
import trees.functions as TF


### Single Learners ###
import trees._classifiers._BinaryDecisionTree
BinaryDecisionTree = \
    trees._classifiers._BinaryDecisionTree._BinaryTreeClassifier


### Ensemble Learners ###
import trees._classifiers._RandomForest
RandomForest = \
    trees._classifiers._RandomForest._RandomForestClassifier

# import trees._classifiers._AdaBoost as _AdaBoost
# AdaBoostedTree = _AdaBoost._AdaBoostedClassifier

# import trees._classifiers._GradientBoost as _GradientBoost
# GradientBoostedTree = _GradientBoost._GradientBoostedClassifier