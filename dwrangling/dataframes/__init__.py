"""Dataframes - Functions to load, manipulate and reconstruct observables as \
event DataFrames."""

import numpy as np
import pandas as pd



### DF Manipulation Functions ###
def merge_dataframes(*dataframes, **kwargs):
    """Merges the :class:`pandas.DataFrame`'s ontop of one another in the order \
    given."""
    return pd.concat(dataframes, ignore_index=True, **kwargs)


def zero_col(dataFrame):
    """Creates a column the same length as the :class:`pandas.DataFrame` but \
    filled with zeros."""
    return pd.DataFrame(0, index=np.arange(len(dataFrame)), columns=['zeros'])


def pick_row_max(column1, column2):
    """Choses the maximum value between two :class:`pandas.DataFrame` columns \
    and returns a single :class:`pandas.DataFrame` column."""

    # Join the dataframes side by side
    joined_df = pd.concat([column1, column2], axis=1)

    # Find the maximum of each row
    result = joined_df.max(axis=1)

    # Convert to a DataFrame and return
    return pd.DataFrame(result, columns=['max'])


def df_move_to_last_col(dataframe, column):
    """Moves the specified column, given as a :class:`str`, to the end of the \
    :class:`pandas.DataFrame`."""
    
    # Obtain list of column names
    col_list = list(dataframe.columns)
    
    # Check column is in the DataFrame
    if not column in col_list:
        raise TypeError("column must be the name of a column in the " +
                        f"DataFrame. You entered '{column}'.")
    
    # Find the position of this column in the DataFrame
    index = col_list.index(column)
    
    # If this column is the last column, do nothing
    if index == len(col_list) - 1:
        return dataframe
    
    # Change ordering of the columns, so given column is last
    col_list = col_list[:index] + col_list[index+1:] + [column,]
    dataframe = dataframe.reindex(columns=col_list)
    
    # Return the reindex-ed DataFrame
    return dataframe


def df_to_ML_input(dataframe):
    """Converts a dataframe to arrays ready for Machine Learning.
    
    Parameters:
     - dataframe: A :class:`pandas.DataFrame`, containing only the observables \
     you wish for ML, with 'signal' as the final column.
    
    
    Returns:
     - observables: A 2d :class:`numpy.ndarray` containing all the observables \
     corresponding to each event from the :class:`pandas.DataFrame`, in the same \
     order as the columns;
     - labels: A 1d :class:`numpy.ndarray` containing all the 'signal' values \
     corresponding to each event's 'observables'."""
    
    # Obtain list of observables.
    input_observables = list(dataframe.columns.values)
   
    # Remove the signal column from observables.
    input_observables.remove('signal')
    
    # Obtain numpy arrays for obervables and signal (labels).
    observables = dataframe[input_observables].values
    labels = dataframe.signal.values
    
    # Return the observables and labels arrays.
    return observables, labels



### Classes ###
class ObservablesDataFrame(pd.DataFrame):
    """A :class:`pandas.DataFrame` which must have a final column of type :class:`bool` named 'signal'."""
    
    def __init__(self, data, *args, columns=None, **kwargs):
        
        # Check columns was given as a parameter
        if isinstance(data, pd.DataFrame):
            columns = list(data.columns)              
        if columns is None:
            raise ValueError("You must enter the column names, " +
                             "in the same order as the data.")
        
        # Check last column is signal
        ObservablesDataFrame.last_col_signal(columns)

        # Create pandas.DataFrame object
        super().__init__(data, *args, columns=columns, **kwargs)
        
        # Check last column contains only 1s and 0s
        if not np.all(np.where((self.signal==0) | (self.signal==1), 1, 0)):
            raise ValueError("The signal columns must be either 0 or 1")


    def get_signal(self):
        """Returns a :class:`pandas.DataFrame` containing only the 'signal' \
        events."""
        return self[self.signal == True].reset_index(drop=True)
    get_s = get_signal

    def get_background(self):
        """Returns a :class:`pandas.DataFrame` containing only the 'background' \
        events."""
        return self[self.signal == False].reset_index(drop=True)
    get_b = get_background
    
    
    def ML_input(self):
        """Returns the dataframe as arrays ready for Machine Learning.
    
    Returns:
     - observables: A 2d :class:`numpy.ndarray` containing all the observables \
     corresponding to each event from the :class:`pandas.DataFrame`, in the same \
     order as the columns;
     - labels: A 1d :class:`numpy.ndarray` containing all the 'signal' values \
     corresponding to each event's 'observables'."""
        return df_to_ML_input(self)
    
    def partition(self, observable, partition_value):
        """Returns the dataframe partitioned at a given value on a given observable.
        
        Paramters:
         - observable: A :class:`str` containing a column name;
         - partition_value: A :class:`float` to partition the dataframe on the given \
         'observable'.
        
        Returns:
         - less_partition: An :class:`ODataFrame` containing the partition with \
         observables less than the 'partition_value';
         - greater_partition: An :class:`ODataFrame` containing the partition with \
         observables greater than or equal to the 'partition_value'.
        """
        
        if not observable in self:
            raise ValueError("Parameter 'observable' must be a 'str' containing a " +
                             f"column name, not '{observable}' of type " +
                             f"'{type(observable)}'.")
        
        mask = self[observable] < partition_value
    
        return ObservablesDataFrame(self[mask]), ObservablesDataFrame(self[~mask])
    
    def copy(self):
        return ObservablesDataFrame(super().copy())


    @staticmethod
    def last_col_signal(columns):
        if isinstance(columns, pd.DataFrame):
            columns = list(columns.columns)
        
        if isinstance(columns, list):
            if columns[-1] != "signal":
                raise ValueError("The last entry for the columns must be " +
                                 f"'signal', not '{columns[-1]}'.")
        else:
            raise ValueError("The columns must be type 'list', " +
                             f"not type '{type(columns)}'.")

        return True


    @classmethod
    def from_sb_dfs(cls, signal_df, background_df, **kwargs):
        """Given a signal and background dataframe, return an ObservablesDataFrame.
        
        Paramters:
         - signal_df: A :class:`numpy.ndarray` with the final column 'signal' \
         containing only type :class:`bool`.
         - background_df: A :class:`numpy.ndarray` with the final column 'signal' \
         containing only type :class:`bool`.
        
        Returns:
         - ODataFrame: An :class:`ObservablesDataFrame` combined from both \
         'signal_df' and 'background_df'."""

        # Ensure dataframes have signal column at end
        cls.last_col_signal(signal_df)
        cls.last_col_signal(background_df)

        return cls(merge_dataframes(signal_df, background_df), **kwargs)
ODataFrame = ObservablesDataFrame