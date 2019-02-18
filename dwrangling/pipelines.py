"""Pipelines - The functions associated with quick data pipelines \
for Machine Learning."""

import dwrangling as DW
import dwrangling.reconstruction as DWR
import dwrangling.events as DWE
import dwrangling.dataframes as DWDF
import dwrangling.lhefiles as DWLHE



def lhe_to_events_pickled(collison, return_events=True):
    f"""Converts LHE files to lists of type 'Events' and pickles them.
    
    Parameters:
     - collision: Must be from {DW.collisions};
     - return_events: 'True' (Default) or 'False', returns events.
    
    Returns:
     - events: As (signal_events, background_events), where each is
               a list of type 'Events'. [Will return 'None' if
               'return_events=False]."""
    
    # Obtain file paths for these events
    paths = DW.get_data_file_paths(collision, '.pkl', 'events')
    
    # Check user doesnt accidently overwrite files
    if os.path.isfile(paths[0]) or os.path.isfile(paths[1]):
        overwrite = DW.y_n_input("These Event files already exist, do you " +
                                 "want to overwrite them? (y/n): ")
        # If user doesnt want to overwrite return False.
        if overwrite is 'n':
            return False
    
    # Convert LHE files to events
    events = DWLHE.lhe_to_events(collision)
    
    # Pickle each list of events
    for event_list, event_path in zip(events, paths):
        DW.pickle_object(event_list, event_path)
    
    # Return list of events if user wants them
    if return_events:
        return events
    
    # Otherwise return 'None'
    return None


def get_events(collision):
    f"""Unpickles the 'Event' files and returns them.
    
    Parameters:
     - collision: Must be from {DW.collisions};
    
    Returns:
     - events: As (signal_events, background_events), where each is
               a list of type 'Events'."""
    
    # Obtain file paths for these events
    paths = DW.get_data_file_paths(collision, '.pkl', 'events')
    
    # Unpickle events
    events = []
    for event_path in paths:
        
        # Check file exists
        if not os.path.isfile(event_path):
            raise IOError(f"No such file or directory: '{event_path}'")
        
        # Add to list of Events
        events.append( DW.unpickle_object(event_path) )
    
    return tuple(events)


def lhe_to_Observables_DataFrames_pickled(collison, return_dataframe=True):
    f"""Converts LHE files to 'ObservablesDataFrame's and pickles them.
    
    Parameters:
     - collision: Must be from {DW.collisions};
     - return_dataframe: 'True' (Default) or 'False', returns a dataframe.
    
    Returns:
     - dataframe: An 'ObservablesDataFrames'. [Will return 'None' if
                  'return_dataframes=False']."""
    
    # Obtain file paths for these events
    # Parse collision and obtain file locations.
    paths = DW.get_data_file_paths(collision, '.pkl', 'dataframes')
    
    # Check user doesnt accidently overwrite files
    if os.path.isfile(paths[0]) or os.path.isfile(paths[1]):
        overwrite = DW.y_n_input("These DataFrame files already exist, do you " +
                                 "want to overwrite them? (y/n): ")
        # If user doesnt want to overwrite return False.
        if overwrite is 'n':
            return False
    
    # Convert LHE files to events
    odfs = DWLHE.lhe_to_dataframes(collision)
    
    # Pickle each list of events
    for odf, event_path in zip(odfs, paths):
        DW.pickle_object(odf, event_path)
    
    # Return list of events if user wants them
    if return_dataframe:
        return DWDF.ODataFrame.from_sb_dfs(*odfs)
    
    # Otherwise return 'None'
    return None


def get_dataframe(collision):
    f"""Unpickles the 'ObservablesDataFrame' files and returns a dataframe.
    
    Parameters:
     - collision: Must be from {DW.collisions};
    
    Returns:
     - dataframe:  An 'ObservablesDataFrames'."""
    
    # Obtain file paths for these events
    paths = DW.get_data_file_paths(collision, '.pkl', 'dataframes')
    
    # Unpickle events
    odfs = []
    for odf_path in paths:
        
        # Check file exists
        if not os.path.isfile(odf_path):
            raise IOError(f"No such file or directory: '{odf_path}'")
        
        # Add to list of ObservablesDataFrames
        odfs.append( DW.unpickle_object(odf_path) )
    
    return DWDF.ODataFrame.from_sb_dfs(*odfs)


def get_ML_dataframe(collision, train_frac = 0.70):
    f"""Unpickles the 'ObservablesDataFrames' files, splits them into training \
and testing sets and returns a dictionary of dataframes - with the optimal \
observables for this collision.
    
    Parameters:
     - collision: Must be from {DW.collisions};
     - train_frac: The approximate fraction of data to use for the training
                   set (Default: 0.7).
    
    Returns:
     - dict: A dictionary of 'ObservablesDataFrames' - with the optimal
             observables for the given collision. The keys are 'train' and
             'test'."""
    
    # Validate train_frac parameter
    if not (0 < train_frac < 1):
        raise ValueError("The training fraction must be between 0 and 1, " +
                         f"not {train_frac}.")
    
    # Retrieve the ObservablesDataFrame for the collision
    dataframe = get_dataframe(collision)
    
    # Get the optimal observables dataframe
    dataframe = DWDF.observables.get_ML_observables_dataframe(collision,
                                                              dataframe)
    
    # Split the dataframe into training and testing sets.
    dict_dataframe = DW.split_data(dataframe, train_frac = train_frac)
    
    # Return the Machine Learning ready dataframe dictionary.
    return dict_dataframe


df_to_ML_input = DW.df_to_ML_input
dataframe_to_ML_input = DW.df_to_ML_input