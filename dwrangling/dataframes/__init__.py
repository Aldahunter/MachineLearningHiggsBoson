"""Dataframes - Functions to load, manipulate and reconstruct observables as \
event DataFrames."""

import numpy as np
import pandas as pd



### DF Manipulation Functions ###
def merge_dataframes(*dataframes, **kwargs):
    """Merges the dataframes ontop of one another in the order given"""
    return pd.concat(dataframes, ignore_index=True, **kwargs)


def zero_col(DataFrame):
    """Creates a column the same length as DataFrame filled with zeros."""
    return pd.DataFrame(0, index=np.arange(len(DataFrame)), columns=['zeros'])


def pick_row_max(dataframe1, dataframe2):
    """Choses the maximum value between 2 columns.

    Parameters:
     - dataframe1: the first column.
     - dataframe2: the second column.

    Returns:
     - restult: a single column DataFrame, with the max value 
                for each row from dataframe1 or dataframe2."""

    # Join the dataframes side by side
    joined_df = pd.concat([dataframe1, dataframe2], axis=1)

    # Find the maximum of each row
    result = joined_df.max(axis=1)

    # Convert to a DataFrame and return
    return pd.DataFrame(result, columns=['max'])


def df_move_to_last_col(dataframe, column):
    """Moves the specified column to the end of the dataframe.
    
    Parameters:
     - dataframe: The DataFrame you want to reindex.
     - column: The name of the column you want to move to the end.
     
    Results:
     - dataframe: The inputted dataframe, with column at the end."""
    
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



### Classes ###
class ObservablesDataFrame(pd.DataFrame):
    
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
        return self[self.signal == True].reset_index(drop=True)
    get_s = get_signal

    def get_background(self):
        return self[self.signal == False].reset_index(drop=True)
    get_b = get_background


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

        # Ensure dataframes have signal column at end
        cls.last_col_signal(signal_df)
        cls.last_col_signal(background_df)

        return cls(merge_dataframes(signal_df, background_df), **kwargs)
ODataFrame = ObservablesDataFrame
        