"""Data Wrangling - Cleaning, Munging, Mapping and Pipelines."""

import os
import pickle
import random
import pandas as pd

import dwrangling.dataframes as DWDF



### Data ###
bs = '//'
comma_space = ', '

#: Valid collision paramters.
collisions = ['pp_2mu2e', 'pp_2mu2nu']

#: Optimal collision observables.
collision_observables = {'pp_2mu2e': ['Z_e_m', 'Z_mu_m',
                                      'delR_e', 'delR_mu', 'delR_Z',
                                      'm_H'],
                         'pp_2mu2nu': []}



### Hidden Functions ###
def _docstring_parameter(*sub, **kwsub):
    """Formats an objects docstring."""
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub, **kwsub)
        return obj
    return dec


def _collision_check(collision):
    collision = collision.strip()
    
    if not collision in collisions:
        raise ValueError(f"'{collision}' is not a valid collision name. " +
                         f"You can choose from: {comma_space.join(collisions)}.")


def _y_n_input(message):
    """Repeatdly ask user message, until they answer yes or no.
    
    Parameters:
     - message: Them message which will be displayed to the user.
    
    Returns:
     - answer: 'y' or 'n', depending on what user answered."""
    
    # Define possible inputs for yes and no
    ys = ['y', 'yeh', 'yes']
    ns = ['n', 'nah', 'no']
    
    # Question user in loop, unitil answer is yes or no
    answer = None
    while (not answer in ys) and (not answer in ns):
        answer = input(message).lower()
    
    # Determine user's response and return 'y' or 'n'
    if answer in ys:
        return 'y'
    else:
        return 'n'



### Pickle Save/Load Functions ###
def pickle_object(object, file_path):
    """Pickles (saves) an 'object' at the location 'file_path'."""
    with open(file_path, 'wb') as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)
    print(f"Successfully pickled '{file_path.split(bs)[-1]}'")

def unpickle_object(file_path):
    """Unpickles (loads) and returns an 'object' at the location 'file_path'."""
    with open(file_path, 'rb') as _input:
          obj = pickle.load(_input)
    return obj



### Machine Learning Functions ###
@_docstring_parameter(doc=DWDF.df_to_ML_input.__doc__)        
def df_to_ML_input(dataframe):
    """{doc}"""
    return DWDF.df_to_ML_input(dataframe)

@_docstring_parameter(collisions=collisions)        
def get_collision_observables(collision):
    """Returns a list of the collision's optimal observables' labels.
    
    Parameters:
     - collision: The desired collision as a :class:`str`, must be in {collisions}.
    
    Returns:
     - list: A :class:`list` of :class:`str`'s corresponding to the collision's \
     optimal observables for Machine Learning."""
    
    
    # Validate and clean inputs
    collision = collision.strip()
    if not collision in collisions:
        raise ValueError(f"'{collision}' is not a valid collision name. " +
                         "You can choose from: " +
                         f"{comma_space.join(collisions)}.")
    
    return collision_observables[collision]


def split_data(dataframe, train_frac = 0.70):
    """Splits the data randomly into a training and testing set.
    
    Parameters:
     - dataframe: The :class:`pandas.DataFrame` you wish to split into sets;
     - train_frac: A :class:`float` giving the fraction of data to use for \
     the training set. [Default: 0.7].
    
    Returns:
     - train: A :class:`dwrangling.dataframes.ODataFrame` for training;
     - test: A :class:`dwrangling.dataframes.ODataFrame` for testing."""
    
    # Validate train_frac parameter
    if not (0 < train_frac < 1):
        raise ValueError("The training fraction must be between 0 and 1, " +
                         f"not {train_frac}.")
    
    # Calculate the train sample size from its fraction
    train_size = int(len(dataframe) * train_frac)
    
    # Randomly sample the train set from the dataframe and order by signal
    train = dataframe.sample(train_size).sort_values('signal')
    
    # Select the complement set from the dataframe for tests
    test = dataframe.loc[[(index not in train.index)
                          for index in dataframe.index]]
    
    # Reset the indexes so go from 0 to N
    test = test.reset_index(level=None, drop=True)
    train = train.reset_index(level=None, drop=True)
    
    # Return as ObservableDataFrames
    return DWDF.ODataFrame(train), DWDF.ODataFrame(test)
    


### File Functions ###
@_docstring_parameter(collisions=collisions)        
def get_data_file_paths(collision, ext, *directories):
    """Returns the file location as a string, for both signal and background.
    
    Parameters:
     - collision: The desired collision as a :class:`str`, must be in {collisions};
     - ext: The file extension/type as a :class:`str`, e.g. 'jpg'.
    
    Returns:
     - signal_file_loc: A :class:`str` containing the path to the signal file;
     - background_file_loc: A :class:`str` containing the path to the background \
     file;"""

    # Validate and clean inputs
    collision, ext = collision.strip(), ext.strip()
    directories = [dir_.strip() for dir_ in directories]
    _collision_check(collision)
    ext = ext if (ext[0] == '.') else '.' + ext
    

    # Get file names for both
    signal_file_name = collision.replace('_', '_h_', 1) + '_heft'
    signal_file_loc = os.path.join(os.getcwd(), "Data", *directories,
                                   signal_file_name + ext)

    background_file_name = collision  + '_bkg'
    background_file_loc = os.path.join(os.getcwd(), "Data", *directories,
                                       background_file_name + ext)

    # Return file locations
    return signal_file_loc, background_file_loc