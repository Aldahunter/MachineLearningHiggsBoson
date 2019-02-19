"""Data Wrangling - Cleaning, Munging, Mapping and Pipelines."""

import os
import pickle

import dwrangling.dataframes as DWDF



### Data ###
bs = '//'
comma_space = ', '
collisions = ['pp_2mu2e', 'pp_2mu2nu']
collision_observables = {'pp_2mu2e': ['m_H', 'Z_mu_rap', 'Z_e_rap', 'mu-_px', 'e-_py'],
                         'pp_2mu2e': []}



### File Functions ###
def collision_check(collision):
    collision = collision.strip()
    
    if not collision in collisions:
        raise ValueError(f"'{collision}' is not a valid collision name. " +
                         f"You can choose from: {comma_space.join(collisions)}.")

def get_data_file_paths(collision, ext, *directories):
    f"""Returns the file location as a string, for both signal and background.
    
    Parameters:
     - collision: The desired collision as a string, must be in {collisions};
     - ext: The file extension/type as a string, e.g. 'jpg'."""

    # Validate and clean inputs
    collision, ext = collision.strip(), ext.strip()
    directories = [dir_.strip() for dir_ in directories]
    collision_check(collision)
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


def y_n_input(message):
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



### Machine Learning Functions ###
def get_collision_observables(collision):
    f"""Returns a list of the collision's optimal observables'labels.
    
    Parameters:
     - collision: The desired collision as a string, must be in {collisions}.
    
    Returns:
     - list: A list of str's corresponding to the collision's optimal \
             observables for Machine Learning."""
    
    
    # Validate and clean inputs
    collision = collision.strip()
    if not collision in collisions:
        raise ValueError(f"'{collision}' is not a valid collision name. " +
                         f"You can choose from: {comma_space.join(collisions)}.")
    
    return collision_observables[collision]


def split_data(dataframe, train_frac = 0.70):
    """Splits the data into a training and testing set.
    
    Parameters:
     - dataframe: The DataFrame you wish to split into sets;
     - train_frac: The approximate fraction of data to use for the training
                   set (Default: 0.7).
    
    Returns:
     - dict: A dictionary containing two 'ObservablesDataFrame's, with the
             keys 'train' and 'test'."""
    
    # Validate train_frac parameter
    if not (0 < train_frac < 1):
        raise ValueError("The training fraction must be between 0 and 1, " +
                         f"not {train_frac}.")
    
    # The data must be a pandas.DataFrame.
    train_data, test_data = [], []

    # Iterate through each row in data.
    for _, datum in data.iterrows():
        
        # If unform random number between [0, 1] is less than,
        # or equal to, the the train_frac.
        if random() <= train_frac:
            # Add row to train_data.
            train_data.append(datum)
        else:
            # If greater than train_frac, add to test_data.
            test_data.append(datum)
    
    # Convert to dataframes
    train_data = pd.DataFrame(train_data).reset_index(drop=True)
    test_data = pd.DataFrame(test_data).reset_index(drop=True)
    
    # Return dictionary of train_data and test_data as panda.DataFrames.
    return {'train': DWDF.ODataFrame(train_data),
            'test': DWDF.ODataFrame(test_data)}


def df_to_ML_input(df):
    """Converts a dataframe to arrays ready for Machine Learning.
    
    Parameters:
     - dataframe: A dataframe, containing only the observables you wish for ML
                  parameters, with 'signal' as the final column.
    
    
    Returns:
     - observables: A 2d-array containing all the observables corresponding to
                    each event from the dataframe, in the same column order;
     - labels: A 1d-array containing all the 'signal' values that correspond to
               each event."""
    
    # Obtain list of observables.
    input_observables = list(df.columns.values)
   
    # Remove the signal column from observables.
    input_observables.remove('signal')
    
    # Obtain numpy arrays for obervables and signal (labels).
    observables = df[input_observables].values
    labels = df.signal.values
    
    # Return the observables and labels arrays.
    return observables, labels



### Pickle Save/Load Functions ###
def pickle_object(object_, file_path):
    """Pickles (saves) an 'object_' at the location 'file_path'."""
    with open(file_path, 'wb') as output:
        pickle.dump(object_, output, pickle.HIGHEST_PROTOCOL)
    print(f"Successfully pickled '{FILE.split(bs)[-1]}'")

def unpickle_object(file_path):
    """Unpickles (loads) and returns an 'object' at the location 'file_path'."""
    with open(file_path, 'rb') as _input:
          obj = pickle.load(_input)
    return obj